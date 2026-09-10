"""CLI. 이 프로젝트의 유일한 진입점이다.

argparse만 쓴다. 외부 CLI 라이브러리를 넣지 않는다
(docs/guidelines/04-interface.md).

여기서 하지 않는 것: 비즈니스 로직, 프롬프트 조립, HTTP 호출. 조립도 직접 하지
않고 composition에 맡긴다.

종료 코드
    0  정상 (거부도 정상이다)
    1  사용자 입력 오류
    2  외부 시스템 오류 (Ollama 연결 실패 등)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src import composition
from src.application.evaluation import CaseOutcome, sample_cases
from src.interface import web
from src.domain.models import Answer, Query, Refusal
from src.domain.policy import DISCLAIMER, RetrievalPolicy
from src.infrastructure.errors import OllamaUnavailable

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_EXTERNAL_ERROR = 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage(sys.stderr)
        print("명령이 필요하다: ingest, ask, eval", file=sys.stderr)
        return EXIT_USER_ERROR

    try:
        if args.command == "ingest":
            return run_ingest(
                source=_source_of(args),
                index_path=_index_of(args),
                resume=args.resume,
                material=args.material,
                limit=args.limit,
            )
        if args.command == "serve":
            return run_serve(port=args.port)
        if args.command == "eval":
            return run_eval(
                qaset=_qaset_of(args),
                index_path=_index_of(args),
                k=args.k,
                min_score=args.min_score,
                sample_size=args.sample,
                seed=args.seed,
                out=args.out,
                change=args.change,
                skip_answers=args.skip_answers,
                material=args.material,
            )
        return run_ask(
            question=args.question,
            index_path=_index_of(args),
            k=args.k,
            min_score=args.min_score,
        )
    except OllamaUnavailable as exc:
        # 서비스 장애는 "근거 없음"과 다른 실패다 (MEMORY.md D-017). 사용자에게
        # 질문을 바꿔 보라고 안내하면 안 된다.
        print(f"Ollama 요청이 실패했다: {exc}", file=sys.stderr)
        print("ollama serve가 떠 있는지 확인하십시오.", file=sys.stderr)
        return EXIT_EXTERNAL_ERROR
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print("색인이 없으면 먼저 ingest를 실행하십시오.", file=sys.stderr)
        return EXIT_USER_ERROR
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USER_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="문제성 피부 메이크업 데이터 RAG 질의응답 (학습용)",
    )
    commands = parser.add_subparsers(dest="command")

    ingest = commands.add_parser("ingest", help="1~4단계: 적재 → 청크 → 임베딩 → 색인")
    ingest.add_argument(
        "--source",
        type=Path,
        default=None,
        help=f"원본 JSON 경로 (기본값: {composition.DEFAULT_SOURCE})",
    )
    ingest.add_argument(
        "--material",
        choices=composition.MATERIALS,
        default=composition.DEFAULT_MATERIAL,
        help=(
            "색인 재료. qa는 질문+답변, excerpt는 근거 발췌, doc은 논문 원문 "
            f"(기본값: {composition.DEFAULT_MATERIAL})"
        ),
    )
    ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="문서 수 상한. --material doc에만 쓴다. 전량은 청크가 30만 개를 넘는다",
    )
    ingest.add_argument(
        "--resume",
        action="store_true",
        help="기존 색인에 이어서 쌓는다. 이미 들어간 청크는 다시 임베딩하지 않는다",
    )
    _add_index_options(ingest)

    ask = commands.add_parser("ask", help="5~6단계: 검색 → 생성")
    ask.add_argument("question", help="질문")
    ask.add_argument(
        "--k", type=int, default=RetrievalPolicy().k, help="검색 상위 몇 개를 볼지"
    )
    _add_index_options(ask)

    server = commands.add_parser("serve", help="로컬 웹 UI를 띄운다 (127.0.0.1)")
    server.add_argument(
        "--port",
        type=int,
        default=web.DEFAULT_PORT,
        help=f"포트 (기본값: {web.DEFAULT_PORT})",
    )

    evaluate = commands.add_parser("eval", help="평가셋으로 채점한다")
    evaluate.add_argument(
        "--qaset",
        type=Path,
        default=None,
        help=f"평가셋 경로 (기본값: {composition.DEFAULT_QASET})",
    )
    _add_index_options(evaluate)
    evaluate.add_argument(
        "--k", type=int, default=RetrievalPolicy().k, help="검색 상위 몇 개를 볼지"
    )
    evaluate.add_argument(
        "--sample",
        type=int,
        default=composition.DEFAULT_SAMPLE_SIZE,
        help=f"채점할 표본 수. 0이면 전량 (기본값: {composition.DEFAULT_SAMPLE_SIZE})",
    )
    evaluate.add_argument(
        "--seed",
        type=int,
        default=composition.SAMPLE_SEED,
        help=f"표본 추출 seed (기본값: {composition.SAMPLE_SEED})",
    )
    evaluate.add_argument(
        "--material",
        choices=composition.MATERIALS,
        default=composition.DEFAULT_MATERIAL,
        help="색인에 무엇이 들어 있는지. 결과 파일의 config에 기록된다",
    )
    evaluate.add_argument("--out", type=Path, default=None, help="결과 JSON을 저장할 경로")
    evaluate.add_argument(
        "--change", default="", help="마지막 평가 이후 바꾼 것 하나 (eval/README.md)"
    )
    evaluate.add_argument(
        "--skip-answers",
        action="store_true",
        help="생성 단계를 건너뛰고 검색만 채점한다 (빠름)",
    )

    return parser


def _add_index_options(command: argparse.ArgumentParser) -> None:
    """색인 경로와 --aihub 플래그. 세 명령이 같은 선택지를 갖는다."""
    command.add_argument(
        "--index",
        type=Path,
        default=None,
        help=f"색인 경로 (기본값: {composition.DEFAULT_INDEX}, --aihub면 {composition.AIHUB_INDEX.name})",
    )
    command.add_argument(
        "--aihub",
        action="store_true",
        help="샘플 대신 AI Hub 실데이터를 쓴다. 경로는 조립부에 있다",
    )
    command.add_argument(
        "--min-score",
        type=float,
        default=RetrievalPolicy().min_score,
        help=f"근거로 인정할 최소 유사도 (기본값: {RetrievalPolicy().min_score})",
    )


def _source_of(args: argparse.Namespace) -> Path:
    if args.source is not None:
        return args.source
    return composition.AIHUB_SOURCE if args.aihub else composition.DEFAULT_SOURCE


def _qaset_of(args: argparse.Namespace) -> Path:
    if args.qaset is not None:
        return args.qaset
    # 실데이터는 QA 라벨링 폴더를 그대로 평가셋으로 쓴다. 중간 변환 파일이 없다.
    return composition.AIHUB_SOURCE if args.aihub else composition.DEFAULT_QASET


def _index_of(args: argparse.Namespace) -> Path:
    if args.index is not None:
        return args.index
    return composition.AIHUB_INDEX if args.aihub else composition.DEFAULT_INDEX


def run_ingest(
    source: Path,
    index_path: Path,
    resume: bool = False,
    material: str = composition.DEFAULT_MATERIAL,
    limit: int | None = None,
) -> int:
    print(f"[1/4] 문서 적재 ... {source} (재료: {material})")
    chunks = composition.build_ingest(source, material=material, limit=limit).execute()
    print(f"[2/4] 청크 분할 ... {len(chunks)}개")

    already = composition.existing_chunk_ids(index_path) if resume else frozenset()
    if resume:
        print(f"[3/4] 임베딩 ... 이미 색인된 {len(already)}개는 건너뛴다")
    else:
        print("[3/4] 임베딩 ...")

    count = composition.build_index_builder(
        resume_from=index_path if resume else None
    ).execute(
        chunks, index_path, on_progress=_print_progress, already_indexed=already
    )

    print(f"[4/4] 색인 저장 ... {index_path} (새로 {count}개)")
    return EXIT_OK


def run_serve(port: int = web.DEFAULT_PORT) -> int:
    """웹 UI를 띄운다. 두 색인을 모두 올려 화면에서 바꿔 볼 수 있게 한다.

    조립은 여기서 하지 않는다. composition이 만든 유스케이스를 넘길 뿐이다.
    """
    answerers = {
        "qa": composition.build_answer_usecase(composition.AIHUB_INDEX),
        "excerpt": composition.build_answer_usecase(composition.AIHUB_EXCERPT_INDEX),
    }
    web.serve(answerers, log_path=composition.HUMAN_CHECK_LOG, port=port)
    return EXIT_OK


def run_ask(
    question: str,
    index_path: Path,
    k: int = RetrievalPolicy().k,
    min_score: float = RetrievalPolicy().min_score,
) -> int:
    if not question.strip():
        print("질문이 비어 있다", file=sys.stderr)
        return EXIT_USER_ERROR

    result = composition.build_answer_usecase(
        index_path, k=k, min_score=min_score
    ).execute(Query(question))

    if isinstance(result, Refusal):
        print(format_refusal(result))
    else:
        print(format_answer(result))
    return EXIT_OK


def run_eval(
    qaset: Path,
    index_path: Path,
    k: int,
    min_score: float,
    sample_size: int,
    seed: int,
    out: Path | None,
    change: str,
    skip_answers: bool,
    material: str = composition.DEFAULT_MATERIAL,
) -> int:
    loaded = composition.build_qaset_loader(qaset).load()
    cases = sample_cases(loaded, size=sample_size, seed=seed)
    print(f"[1/2] 검색 채점 ... 케이스 {len(cases)}건 (전체 {len(loaded)}건, seed {seed})")

    retrieval = composition.build_retrieval_eval(index_path, k).execute(
        cases, on_progress=_print_progress
    )
    payload: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "change": change,
        **retrieval.to_dict(),
    }
    print(
        f"      Recall@{k}: {retrieval.recall:.3f} "
        f"(채점 {retrieval.graded}건 중 {retrieval.hits}건 적중, 전체 {retrieval.total}건)"
    )
    print(f"      평균 최고 유사도: {retrieval.mean_best_score:.3f}")

    if skip_answers:
        print("[2/2] 답변 채점 생략 (--skip-answers)")
    else:
        print("[2/2] 답변 채점 ... 케이스마다 생성이 일어난다")
        answers = composition.build_answer_eval(
            index_path, k=k, min_score=min_score
        ).execute(
            cases, on_progress=_print_progress
        )
        payload.update(answers.to_dict())
        print(
            f"      거부율: {answers.refusal_rate:.3f} "
            f"(답변 {answers.answered}건, 거부 {answers.refused}건 {answers.by_reason})"
        )
        if answers.failed:
            print(f"      생성 실패 {answers.failed}건 (거부율 분모에서 제외)")
        print(
            f"      답변 정확도(음절 2-gram F1): {answers.accuracy:.3f} "
            f"(채점 {answers.scored}건)"
        )
        if answers.similarity_scores is not None:
            print(f"      보조 지표 임베딩 코사인: {answers.similarity:.3f}")

    payload["config"] = composition.eval_config(
        k,
        sample_size=sample_size,
        seed=seed,
        min_score=min_score,
        material=material,
        index_path=index_path,
    )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"저장: {out}")
        if not skip_answers:
            print(f"저장: {_write_case_rows(out, answers.cases)}")

    return EXIT_OK


def _write_case_rows(out: Path, cases: tuple[CaseOutcome, ...]) -> Path:
    """케이스별 결과를 out 옆의 raw/ 아래에 쓴다.

    실험끼리 공통 케이스만 골라 비교하려면 이 파일이 있어야 한다 (미해결 O-5).
    `eval/raw/`는 .gitignore 대상이다. 다만 이 파일에도 질문과 답변 원문은 들어
    있지 않다. 케이스 ID와 점수뿐이다 (SAFETY.md S-7).
    """
    path = out.parent / "raw" / f"{out.stem}.cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(row) for row in cases], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def format_answer(answer: Answer) -> str:
    """답변 본문 + 근거 + 면책 문구.

    셋 중 하나라도 빠지면 안 된다. 근거를 보여 주는 것이 이 시스템의 존재
    이유다 (docs/guidelines/04-interface.md).
    """
    passages = "\n\n".join(
        f"  [{number}] {item.chunk.id} (유사도 {item.score:.2f}{_meta(item.chunk)})\n"
        f"      {item.chunk.text}"
        for number, item in enumerate(answer.evidence, start=1)
    )
    return f"{answer.body}\n\n근거 {len(answer.evidence)}건:\n{passages}\n\n{DISCLAIMER}"


def format_refusal(refusal: Refusal) -> str:
    """거부 사유를 그대로 보여 준다. 사과 문구를 덧붙이지 않는다."""
    return f"답할 수 없습니다.\n사유: {refusal.reason.value} — {refusal.detail}"


def _meta(chunk: object) -> str:
    parts = [
        value.value if hasattr(value, "value") else value
        for value in (getattr(chunk, "skin_type", None), getattr(chunk, "area", None))
        if value is not None
    ]
    return f", {' · '.join(str(p) for p in parts)}" if parts else ""


def _print_progress(done: int, total: int) -> None:
    # 실데이터는 9,377건이라 오래 걸린다. 진행이 보여야 멈춘 것과 구분된다.
    print(f"      {done} / {total}")


if __name__ == "__main__":  # pragma: no cover - python -m src.interface.cli 대비
    raise SystemExit(main())
