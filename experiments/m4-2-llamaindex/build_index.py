"""M4-2 · LlamaIndex로 1~4단계를 다시 만든다 (적재 → 청크 → 임베딩 → 색인).

**이 폴더의 코드는 제품이 아니다.** 비교 실험이다. `MEMORY.md` D-003은
LlamaIndex를 도입하지 않기로 했고, `docs/PACKAGES.md`가 "비교 목적으로만"이라는
예외를 뒀다. 그래서 별도 가상환경(`.venv-llamaindex`)에서만 돌아간다. 메인
`requirements.txt`에는 아무것도 추가하지 않았다.

**비교가 성립하려면 조건이 같아야 한다.** 다음을 우리 구현과 똑같이 맞췄다.

| 항목 | 값 | 근거 |
|---|---|---|
| 색인 재료 | QA쌍 (질문 + 답변) | D-032 |
| 청킹 | 쪼개지 않는다 | D-032, E-004 |
| 임베딩 모델 | bge-m3 | D-005 |
| 평가 200건 | 색인에서 제외 | 누출 방지 |

1단계(적재)는 우리 로더를 그대로 쓴다. AI Hub JSON은 구조가 고유해서 어느
프레임워크를 쓰든 파싱 코드를 직접 짜야 한다. 여기서 프레임워크가 절약해 주는
것은 없다. 그래서 비교 대상에서 뺐다.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

from src.application.evaluation import sample_cases
from src.infrastructure.aihub_loader import AihubQaPairLoader, AihubQaSetLoader

SOURCE = PROJECT_ROOT / "data" / "raw"
STORE = PROJECT_ROOT / "index" / "llamaindex"

EMBEDDING_MODEL = "bge-m3"
SAMPLE_SIZE = 200
SAMPLE_SEED = 42

# QA쌍 평균이 855자다. 8,192자면 쪼개지지 않는다. LlamaIndex는 청커를 끄는 방법을
# 따로 주지 않아서 한도를 크게 잡는 방식으로 같은 효과를 낸다.
CHUNK_SIZE = 8192


def excluded_ids() -> set[str]:
    """평가에 쓸 200건의 ID. 색인에 넣으면 답을 그대로 베낀다."""
    cases = AihubQaSetLoader(SOURCE).load()
    return {case.id for case in sample_cases(cases, size=SAMPLE_SIZE, seed=SAMPLE_SEED)}


def documents(skip: set[str]) -> list[Document]:
    loader = AihubQaPairLoader(SOURCE)
    rows = [
        Document(text=document.text, doc_id=document.id)
        for document in loader.load()
        if document.id not in skip
    ]
    if loader.errors:
        print(f"    건너뛴 파일 {len(loader.errors)}건")
    return rows


def main() -> int:
    print("[1/4] 문서 적재 ...")
    skip = excluded_ids()
    rows = documents(skip)
    print(f"    QA쌍 {len(rows)}건 (평가용 {len(skip)}건 제외)")

    print("[2/4] 청크 분할 ...")
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    nodes = splitter.get_nodes_from_documents(rows)
    print(f"    노드 {len(nodes)}개")

    print(f"[3/4] 임베딩 + [4/4] 색인 ... ({EMBEDDING_MODEL})")
    started = time.perf_counter()
    index = VectorStoreIndex(
        nodes,
        embed_model=OllamaEmbedding(model_name=EMBEDDING_MODEL),
        show_progress=True,
    )
    elapsed = time.perf_counter() - started

    STORE.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORE))
    print(f"    저장: {STORE} ({elapsed / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
