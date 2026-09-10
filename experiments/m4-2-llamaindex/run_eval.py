"""M4-2 · LlamaIndex로 5~6단계를 돌리고 같은 표본으로 채점한다 (검색 → 생성).

채점 기준은 우리 것을 그대로 쓴다. `bigram_f1`(D-026), 표본 200건, seed 42.
지표 계산까지 프레임워크에 맡기면 점수 차이가 파이프라인 때문인지 채점 방식
때문인지 알 수 없다.

**안전 게이트도 우리 것을 쓴다.** 유사도 임계값(생성 전)과 `SafetyPolicy`
(생성 후)는 도메인 규칙이지 프레임워크 기능이 아니다. 이것까지 바꾸면 비교
대상이 흐려진다. LlamaIndex에 맡긴 것은 검색과 생성 배선뿐이다.

비교하려는 것은 세 가지다.

1. 같은 조건에서 점수가 같이 나오는가 (파이프라인이 동등한가)
2. 코드가 얼마나 줄어드는가
3. 무엇이 보이지 않게 되는가
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from src.application.answering import INSTRUCTION, NO_ANSWER_PATTERN, NO_ANSWER_TOKEN
from src.application.evaluation import sample_cases
from src.domain.policy import SafetyPolicy
from src.domain.scoring import bigram_f1
from src.infrastructure.aihub_loader import AihubQaSetLoader

SOURCE = PROJECT_ROOT / "data" / "raw"
STORE = PROJECT_ROOT / "index" / "llamaindex"
OUT = PROJECT_ROOT / "eval" / "m4-2-llamaindex.json"
CASES_OUT = PROJECT_ROOT / "eval" / "raw" / "m4-2-llamaindex.cases.json"

EMBEDDING_MODEL = "bge-m3"
GENERATION_MODEL = "qwen2.5:3b"
TOP_K = 3
MIN_SCORE = 0.35
NUM_CTX = 8192
TIMEOUT = 300.0
SAMPLE_SIZE = 200
SAMPLE_SEED = 42


def build_prompt(question: str, evidence: list[str]) -> str:
    """우리 `AnswerQuestion.build_prompt`와 같은 모양으로 만든다."""
    blocks = "\n\n".join(f"[근거 {i + 1}]\n{text}" for i, text in enumerate(evidence))
    return f"{INSTRUCTION}\n\n{blocks}\n\n질문: {question}"


def main() -> int:
    print("[준비] 색인 적재 ...")
    embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    storage = StorageContext.from_defaults(persist_dir=str(STORE))
    index = load_index_from_storage(storage, embed_model=embed_model)

    retriever = index.as_retriever(similarity_top_k=TOP_K)
    cutoff = SimilarityPostprocessor(similarity_cutoff=MIN_SCORE)
    llm = Ollama(
        model=GENERATION_MODEL,
        temperature=0.0,
        request_timeout=TIMEOUT,
        context_window=NUM_CTX,
        additional_kwargs={"num_ctx": NUM_CTX},
    )
    safety = SafetyPolicy.default()

    cases = sample_cases(
        AihubQaSetLoader(SOURCE).load(), size=SAMPLE_SIZE, seed=SAMPLE_SEED
    )
    print(f"[평가] 케이스 {len(cases)}건")

    rows: list[dict[str, object]] = []
    scores: list[float] = []
    latencies: list[float] = []
    reasons: dict[str, int] = {}
    failed = 0

    for number, case in enumerate(cases, start=1):
        started = time.perf_counter()
        try:
            found = cutoff.postprocess_nodes(retriever.retrieve(case.question))
            if not found:
                outcome, detail, accuracy = "refused", "근거 없음", None
                reasons["근거 없음"] = reasons.get("근거 없음", 0) + 1
            else:
                prompt = build_prompt(
                    case.question, [node.get_content() for node in found]
                )
                body = str(llm.complete(prompt)).strip()

                violated = safety.violation(body)
                if NO_ANSWER_TOKEN in body or NO_ANSWER_PATTERN.search(body):
                    outcome, detail, accuracy = "refused", "모델이 답을 못 만듦", None
                    reasons[detail] = reasons.get(detail, 0) + 1
                elif violated is not None:
                    outcome, detail, accuracy = "refused", violated, None
                    reasons[violated] = reasons.get(violated, 0) + 1
                else:
                    outcome, detail = "answered", None
                    accuracy = bigram_f1(body, case.expected)
                    scores.append(accuracy)
        except Exception as error:  # noqa: BLE001 - 케이스 하나로 전체를 멈추지 않는다
            outcome, detail, accuracy = "failed", type(error).__name__, None
            failed += 1

        latencies.append(time.perf_counter() - started)
        rows.append(
            {
                "case_id": case.id,
                "outcome": outcome,
                "accuracy": accuracy,
                "detail": detail,
            }
        )
        if number % 10 == 0:
            print(f"      {number} / {len(cases)}", flush=True)

    answered = sum(1 for row in rows if row["outcome"] == "answered")
    refused = sum(1 for row in rows if row["outcome"] == "refused")
    latencies.sort()

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "change": "M4-2 LlamaIndex로 재구현 (같은 재료·모델·k·임계값)",
        "sample_count": len(cases),
        "answer_accuracy": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "scored_count": len(scores),
        "answered_count": answered,
        "refused_count": refused,
        "refusal_rate": round(refused / max(answered + refused, 1), 4),
        "refusal_reasons": reasons,
        "failed_count": failed,
        "answer_latency_p50": round(latencies[len(latencies) // 2], 2),
        "config": {
            "framework": "llama-index-core 0.14.24",
            "chunk_size": 8192,
            "chunk_overlap": 0,
            "top_k": TOP_K,
            "min_score": MIN_SCORE,
            "embed_model": EMBEDDING_MODEL,
            "gen_model": GENERATION_MODEL,
            "material": "qa",
            "sample_size": SAMPLE_SIZE,
            "seed": SAMPLE_SEED,
            "index": "llamaindex/",
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CASES_OUT.parent.mkdir(parents=True, exist_ok=True)
    CASES_OUT.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"      거부 {refused}건 {reasons}")
    print(f"      생성 실패 {failed}건")
    print(
        f"      답변 정확도(음절 2-gram F1): {payload['answer_accuracy']} "
        f"(채점 {len(scores)}건)"
    )
    print(f"      응답 시간 중앙값 {payload['answer_latency_p50']}초")
    print(f"저장: {OUT}")
    print(f"저장: {CASES_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
