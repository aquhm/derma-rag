"""CLI 스모크 테스트.

로직 검증은 application 테스트에서 이미 했다. 여기서는 명령이 끝까지 도는지,
출력에 빠진 것이 없는지, 종료 코드가 맞는지만 본다
(docs/guidelines/04-interface.md).

Ollama가 필요한 경로는 조립부를 대역으로 바꿔 끊는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import composition
from src.application.evaluation import (
    AnswerReport,
    CaseOutcome,
    CaseResult,
    RetrievalReport,
)
from src.domain.models import (
    Answer,
    Chunk,
    EvalCase,
    Evidence,
    Query,
    Refusal,
    RefusalReason,
    SkinType,
)
from src.domain.policy import DISCLAIMER, RetrievalPolicy
from src.infrastructure.errors import EmbeddingUnavailable
from src.interface import cli


def evidence(score: float = 0.87) -> Evidence:
    return Evidence(
        chunk=Chunk(
            id="sample-001#0",
            doc_id="sample-001",
            text="민감성 피부는 자극에 빠르게 반응한다고 기술된다.",
            skin_type=SkinType.SENSITIVE,
            area="볼",
        ),
        score=score,
    )


class FakeAnswerUsecase:
    def __init__(self, result: Answer | Refusal | Exception) -> None:
        self._result = result
        self.questions: list[str] = []

    def execute(self, query: Query) -> Answer | Refusal:
        self.questions.append(query.text)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class FakeIngestUsecase:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def execute(self) -> list[Chunk]:
        return self._chunks


class FakeIndexBuilder:
    def __init__(self) -> None:
        self.saved: list[Path] = []
        self.skipped: list[frozenset[str]] = []

    def execute(  # type: ignore[no-untyped-def]
        self, chunks, path, on_progress=None, already_indexed=frozenset()
    ) -> int:
        if on_progress is not None:
            on_progress(len(chunks), len(chunks))
        self.saved.append(path)
        self.skipped.append(already_indexed)
        return len(chunks)


def use_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    answer: FakeAnswerUsecase | None = None,
    ingest: FakeIngestUsecase | None = None,
    builder: FakeIndexBuilder | None = None,
) -> None:
    if answer is not None:
        monkeypatch.setattr(
            composition,
            "build_answer_usecase",
            lambda path, k=5, min_score=0.35: answer,
        )
    if ingest is not None:
        monkeypatch.setattr(
            composition, "build_ingest", lambda source, material="qa", limit=None: ingest
        )
    if builder is not None:
        monkeypatch.setattr(
            composition, "build_index_builder", lambda resume_from=None: builder
        )


# ------------------------------------------------------------------ 출력 형식

def test_answer_output_has_body_evidence_and_disclaimer() -> None:
    answer = Answer(body="자료에는 향료를 피하라고 기술돼 있습니다.", evidence=(evidence(),))

    printed = cli.format_answer(answer)

    assert "자료에는 향료를 피하라고 기술돼 있습니다." in printed
    assert "sample-001#0" in printed
    assert "민감성 피부는 자극에 빠르게 반응한다고 기술된다." in printed
    assert printed.rstrip().endswith(DISCLAIMER)


def test_answer_output_shows_the_similarity_score() -> None:
    printed = cli.format_answer(Answer(body="본문", evidence=(evidence(0.87),)))

    assert "0.87" in printed


def test_refusal_output_shows_the_reason_without_apology() -> None:
    refusal = Refusal(
        reason=RefusalReason.NO_EVIDENCE, detail="최고 유사도 0.21 < 임계값 0.35"
    )

    printed = cli.format_refusal(refusal)

    assert "답할 수 없습니다" in printed
    assert "최고 유사도 0.21 < 임계값 0.35" in printed
    assert "죄송" not in printed


# --------------------------------------------------------------------- ask

def test_ask_prints_the_answer_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    usecase = FakeAnswerUsecase(Answer(body="본문", evidence=(evidence(),)))
    use_fakes(monkeypatch, answer=usecase)

    code = cli.main(["ask", "민감성 피부 질문"])

    assert code == 0
    assert usecase.questions == ["민감성 피부 질문"]
    assert DISCLAIMER in capsys.readouterr().out


def test_ask_prints_the_refusal_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 거부는 정상 동작이다. 오류 코드로 돌려주면 스크립트가 실패로 읽는다.
    use_fakes(
        monkeypatch,
        answer=FakeAnswerUsecase(
            Refusal(reason=RefusalReason.NO_EVIDENCE, detail="검색 결과가 없다")
        ),
    )

    code = cli.main(["ask", "무관한 질문"])

    assert code == 0
    assert "답할 수 없습니다" in capsys.readouterr().out


def test_blank_question_is_a_user_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fakes(monkeypatch, answer=FakeAnswerUsecase(Answer(body="본문", evidence=(evidence(),))))

    code = cli.main(["ask", "   "])

    assert code == 1
    assert "질문" in capsys.readouterr().err


def test_missing_index_is_a_user_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(path: Path, k: int = 5, min_score: float = 0.35) -> FakeAnswerUsecase:
        raise FileNotFoundError(f"색인 파일이 없다: {path}")

    monkeypatch.setattr(composition, "build_answer_usecase", missing)

    code = cli.main(["ask", "질문"])

    assert code == 1
    assert "ingest" in capsys.readouterr().err


def test_ollama_failure_exits_with_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fakes(monkeypatch, answer=FakeAnswerUsecase(EmbeddingUnavailable("연결 실패")))

    code = cli.main(["ask", "질문"])

    assert code == 2
    assert "Ollama" in capsys.readouterr().err


def test_ask_uses_the_given_index_path(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def record(path: Path, k: int = 5, min_score: float = 0.35) -> FakeAnswerUsecase:
        seen.append(path)
        return FakeAnswerUsecase(Answer(body="본문", evidence=(evidence(),)))

    monkeypatch.setattr(composition, "build_answer_usecase", record)

    cli.main(["ask", "질문", "--index", "index/다른색인.npz"])

    assert seen == [Path("index/다른색인.npz")]


# ------------------------------------------------------------------ ingest

def test_ingest_reports_counts_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    chunks = [Chunk(id=f"d1#{i}", doc_id="d1", text="문단") for i in range(3)]
    builder = FakeIndexBuilder()
    use_fakes(monkeypatch, ingest=FakeIngestUsecase(chunks), builder=builder)

    code = cli.main(["ingest", "--index", str(tmp_path / "index.npz")])

    assert code == 0
    assert builder.saved == [tmp_path / "index.npz"]
    out = capsys.readouterr().out
    assert "3" in out


def test_ingest_reports_a_missing_source_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(source: Path, material: str = "qa", limit: int | None = None) -> FakeIngestUsecase:
        raise FileNotFoundError(f"파일이 없다: {source}")

    monkeypatch.setattr(composition, "build_ingest", missing)

    code = cli.main(["ingest", "--source", "없는파일.json"])

    assert code == 1
    assert "없는파일.json" in capsys.readouterr().err


def test_ingest_uses_default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    sources: list[Path] = []
    builder = FakeIndexBuilder()

    def record(source: Path, material: str = "qa", limit: int | None = None) -> FakeIngestUsecase:
        sources.append(source)
        return FakeIngestUsecase([Chunk(id="d1#0", doc_id="d1", text="문단")])

    monkeypatch.setattr(composition, "build_ingest", record)
    monkeypatch.setattr(
            composition, "build_index_builder", lambda resume_from=None: builder
        )

    cli.main(["ingest"])

    assert sources == [composition.DEFAULT_SOURCE]
    assert builder.saved == [composition.DEFAULT_INDEX]


# -------------------------------------------------------------------- 인자

def test_no_command_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main([])

    assert code == 1
    assert "ingest" in capsys.readouterr().err


def test_ask_without_a_question_is_an_argparse_error() -> None:
    with pytest.raises(SystemExit):
        cli.main(["ask"])


# --------------------------------------------------------------------- eval

class FakeQaSetLoader:
    def __init__(self, cases: list) -> None:  # type: ignore[type-arg]
        self._cases = cases

    def load(self) -> list:  # type: ignore[type-arg]
        return self._cases


class FakeRetrievalEval:
    def __init__(self, report) -> None:  # type: ignore[no-untyped-def]
        self._report = report
        self.calls = 0

    def execute(self, cases, on_progress=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        if on_progress is not None:
            on_progress(len(cases), len(cases))
        return self._report


class FakeAnswerEval:
    def __init__(self, report) -> None:  # type: ignore[no-untyped-def]
        self._report = report
        self.calls = 0

    def execute(self, cases, on_progress=None):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self._report


def eval_cases() -> list[EvalCase]:
    return [
        EvalCase(id="Q-1", question="질문 1", gold_doc_ids=("sample-001",)),
        EvalCase(id="Q-2", question="질문 2", gold_doc_ids=("sample-002",)),
    ]


def retrieval_report() -> RetrievalReport:
    return RetrievalReport(
        k=5,
        results=(
            CaseResult(case_id="Q-1", hit=True, best_score=0.9, retrieved_doc_ids=("sample-001",)),
            CaseResult(case_id="Q-2", hit=False, best_score=0.4, retrieved_doc_ids=("sample-009",)),
        ),
    )


def answer_report() -> AnswerReport:
    return AnswerReport(total=2, answered=1, by_reason={"근거 없음": 1})


def use_eval_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retrieval: FakeRetrievalEval | None = None,
    answers: FakeAnswerEval | None = None,
    cases: list[EvalCase] | None = None,
) -> None:
    monkeypatch.setattr(
        composition, "build_qaset_loader", lambda path: FakeQaSetLoader(cases or eval_cases())
    )
    monkeypatch.setattr(
        composition,
        "build_retrieval_eval",
        lambda index_path, k: retrieval or FakeRetrievalEval(retrieval_report()),
    )
    monkeypatch.setattr(
        composition,
        "build_answer_eval",
        lambda index_path, k=5, min_score=0.35: answers or FakeAnswerEval(answer_report()),
    )


def test_eval_prints_recall_and_refusal_rate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_eval_fakes(monkeypatch)

    code = cli.main(["eval"])

    out = capsys.readouterr().out
    assert code == 0
    assert "Recall@3" in out
    assert "0.5" in out
    assert "거부율" in out


def test_eval_can_skip_the_answer_stage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 생성은 케이스당 수 초가 걸린다. 검색만 재는 실행이 필요하다 (M3-1, M3-3).
    answers = FakeAnswerEval(answer_report())
    use_eval_fakes(monkeypatch, answers=answers)

    code = cli.main(["eval", "--skip-answers"])

    assert code == 0
    assert answers.calls == 0
    assert "거부율" not in capsys.readouterr().out


def test_eval_writes_a_result_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_eval_fakes(monkeypatch)
    out = tmp_path / "result.json"

    cli.main(["eval", "--out", str(out), "--change", "청크 512→256"])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["recall_at_5"] == 0.5
    assert payload["refusal_rate"] == 0.5
    assert payload["change"] == "청크 512→256"
    assert payload["timestamp"]
    assert payload["config"]["top_k"] == 3


def test_eval_result_file_carries_the_model_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 설정을 함께 저장하지 않으면 나중에 어떤 조건의 점수인지 알 수 없다
    # (eval/README.md).
    use_eval_fakes(monkeypatch)
    out = tmp_path / "result.json"

    cli.main(["eval", "--out", str(out)])

    config = json.loads(out.read_text(encoding="utf-8"))["config"]
    assert config["embed_model"] == composition.EMBEDDING_MODEL
    assert config["gen_model"] == composition.GENERATION_MODEL
    assert config["min_score"] == RetrievalPolicy().min_score


def test_eval_result_file_has_answer_accuracy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # M2-7에서 채점 방식을 정했다 (MEMORY.md D-026). 음절 2-gram F1이다.
    use_eval_fakes(monkeypatch)
    out = tmp_path / "result.json"

    cli.main(["eval", "--out", str(out)])

    assert "answer_accuracy" in json.loads(out.read_text(encoding="utf-8"))


def test_eval_passes_k_to_the_usecase(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []
    monkeypatch.setattr(
        composition, "build_qaset_loader", lambda path: FakeQaSetLoader(eval_cases())
    )
    monkeypatch.setattr(
        composition,
        "build_retrieval_eval",
        lambda index_path, k: seen.append(k) or FakeRetrievalEval(retrieval_report()),
    )
    monkeypatch.setattr(
        composition,
        "build_answer_eval",
        lambda index_path, k=5, min_score=0.35: FakeAnswerEval(answer_report()),
    )

    cli.main(["eval", "--k", "3", "--skip-answers"])

    assert seen == [3]


def test_eval_reports_a_missing_qaset_file(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(path: Path) -> FakeQaSetLoader:
        raise FileNotFoundError(f"평가셋 파일이 없다: {path}")

    monkeypatch.setattr(composition, "build_qaset_loader", missing)

    code = cli.main(["eval", "--qaset", "없는파일.json"])

    assert code == 1
    assert "없는파일.json" in capsys.readouterr().err


# ----------------------------------------------------- 실데이터 (--aihub)

def test_aihub_flag_selects_the_dataset_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    # 명령줄에 원본 데이터 경로를 적으면 guard_data.py 훅이 막는다. 플래그로 고른다.
    sources: list[Path] = []
    builder = FakeIndexBuilder()

    def record(source: Path, material: str = "qa", limit: int | None = None) -> FakeIngestUsecase:
        sources.append(source)
        return FakeIngestUsecase([Chunk(id="K-1#0", doc_id="K-1", text="문단")])

    monkeypatch.setattr(composition, "build_ingest", record)
    monkeypatch.setattr(
            composition, "build_index_builder", lambda resume_from=None: builder
        )

    cli.main(["ingest", "--aihub"])

    assert sources == [composition.AIHUB_SOURCE]
    assert builder.saved == [composition.AIHUB_INDEX]


def test_explicit_index_wins_over_the_aihub_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    builder = FakeIndexBuilder()
    monkeypatch.setattr(
        composition,
        "build_ingest",
        lambda source, material="qa", limit=None: FakeIngestUsecase(
            [Chunk(id="K-1#0", doc_id="K-1", text="문단")]
        ),
    )
    monkeypatch.setattr(
            composition, "build_index_builder", lambda resume_from=None: builder
        )

    cli.main(["ingest", "--aihub", "--index", str(tmp_path / "다른.npz")])

    assert builder.saved == [tmp_path / "다른.npz"]


def test_ask_can_use_the_aihub_index(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def record(path: Path, k: int = 5, min_score: float = 0.35) -> FakeAnswerUsecase:
        seen.append(path)
        return FakeAnswerUsecase(Answer(body="본문", evidence=(evidence(),)))

    monkeypatch.setattr(composition, "build_answer_usecase", record)

    cli.main(["ask", "질문", "--aihub"])

    assert seen == [composition.AIHUB_INDEX]


def test_eval_aihub_uses_the_dataset_qaset_and_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qasets: list[Path] = []
    indexes: list[Path] = []

    monkeypatch.setattr(
        composition,
        "build_qaset_loader",
        lambda path: qasets.append(path) or FakeQaSetLoader(eval_cases()),
    )
    monkeypatch.setattr(
        composition,
        "build_retrieval_eval",
        lambda index_path, k: indexes.append(index_path)
        or FakeRetrievalEval(retrieval_report()),
    )

    cli.main(["eval", "--aihub", "--skip-answers"])

    assert qasets == [composition.AIHUB_SOURCE]
    assert indexes == [composition.AIHUB_INDEX]


def test_eval_prints_answer_accuracy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = AnswerReport(
        total=2, answered=2, by_reason={}, accuracy_scores=(1.0, 0.5)
    )
    use_eval_fakes(monkeypatch, answers=FakeAnswerEval(report))

    cli.main(["eval"])

    out = capsys.readouterr().out
    assert "정확도" in out
    assert "0.750" in out


def test_eval_result_file_carries_accuracy_and_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = AnswerReport(total=2, answered=2, by_reason={}, accuracy_scores=(1.0, 0.5))
    use_eval_fakes(monkeypatch, answers=FakeAnswerEval(report))
    out = tmp_path / "result.json"

    cli.main(["eval", "--out", str(out), "--sample", "2", "--seed", "7"])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["answer_accuracy"] == 0.75
    assert payload["config"]["sample_size"] == 2
    assert payload["config"]["seed"] == 7


def test_eval_samples_the_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    # 표본을 정해 두지 않으면 실험마다 다른 케이스를 채점하게 된다 (D-026).
    many = [
        EvalCase(id=f"Q-{i}", question=f"질문 {i}", gold_doc_ids=("d1",))
        for i in range(50)
    ]
    seen: list[int] = []

    class CountingEval(FakeRetrievalEval):
        def execute(self, cases, on_progress=None):  # type: ignore[no-untyped-def]
            seen.append(len(cases))
            return retrieval_report()

    use_eval_fakes(monkeypatch, cases=many, retrieval=CountingEval(retrieval_report()))

    cli.main(["eval", "--sample", "10", "--skip-answers"])

    assert seen == [10]


def test_eval_sample_zero_uses_every_case(monkeypatch: pytest.MonkeyPatch) -> None:
    many = [
        EvalCase(id=f"Q-{i}", question=f"질문 {i}", gold_doc_ids=("d1",))
        for i in range(50)
    ]
    seen: list[int] = []

    class CountingEval(FakeRetrievalEval):
        def execute(self, cases, on_progress=None):  # type: ignore[no-untyped-def]
            seen.append(len(cases))
            return retrieval_report()

    use_eval_fakes(monkeypatch, cases=many, retrieval=CountingEval(retrieval_report()))

    cli.main(["eval", "--sample", "0", "--skip-answers"])

    assert seen == [50]


# ------------------------------------------------------------- 재개 (M2-4)

def test_resume_passes_existing_chunk_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    # 10분짜리 적재가 끊겼을 때 이미 넣은 것을 다시 임베딩하지 않는다.
    builder = FakeIndexBuilder()
    monkeypatch.setattr(
        composition,
        "build_ingest",
        lambda source, material="qa", limit=None: FakeIngestUsecase(
            [Chunk(id="K-1#0", doc_id="K-1", text="문단")]
        ),
    )
    monkeypatch.setattr(composition, "build_index_builder", lambda resume_from=None: builder)
    monkeypatch.setattr(
        composition, "existing_chunk_ids", lambda path: frozenset({"K-1#0"})
    )

    cli.main(["ingest", "--resume"])

    assert builder.skipped == [frozenset({"K-1#0"})]


def test_without_resume_nothing_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = FakeIndexBuilder()
    monkeypatch.setattr(
        composition,
        "build_ingest",
        lambda source, material="qa", limit=None: FakeIngestUsecase(
            [Chunk(id="K-1#0", doc_id="K-1", text="문단")]
        ),
    )
    monkeypatch.setattr(composition, "build_index_builder", lambda resume_from=None: builder)
    monkeypatch.setattr(
        composition,
        "existing_chunk_ids",
        lambda path: pytest.fail("재개가 아니면 기존 색인을 읽지 않는다"),
    )

    cli.main(["ingest"])

    assert builder.skipped == [frozenset()]


# ------------------------------------------- 케이스별 결과 파일 (O-5)

def test_eval_writes_case_rows_next_to_the_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = AnswerReport(
        total=2,
        answered=1,
        by_reason={"근거 없음": 1},
        accuracy_scores=(0.5,),
        cases=(
            CaseOutcome("Q-1", "answered", accuracy=0.5),
            CaseOutcome("Q-2", "refused", detail="근거 없음"),
        ),
    )
    use_eval_fakes(monkeypatch, answers=FakeAnswerEval(report))
    out = tmp_path / "실험.json"

    cli.main(["eval", "--out", str(out)])

    rows = json.loads((tmp_path / "raw" / "실험.cases.json").read_text(encoding="utf-8"))
    assert [r["case_id"] for r in rows] == ["Q-1", "Q-2"]
    assert rows[0]["accuracy"] == 0.5
    assert rows[1]["outcome"] == "refused"


def test_case_rows_are_not_written_without_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_eval_fakes(monkeypatch)

    cli.main(["eval"])

    assert not (tmp_path / "raw").exists()


def test_case_rows_are_skipped_when_answers_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_eval_fakes(monkeypatch)
    out = tmp_path / "실험.json"

    cli.main(["eval", "--out", str(out), "--skip-answers"])

    assert not (tmp_path / "raw" / "실험.cases.json").exists()


# --------------------------------------------- 검색 정책 조정 (M3-3)

def test_eval_passes_k_and_min_score_to_the_answer_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --k가 검색 채점에만 걸리고 답변 경로에 안 걸리면 실험이 성립하지 않는다.
    seen: list[tuple[int, float]] = []

    monkeypatch.setattr(
        composition, "build_qaset_loader", lambda path: FakeQaSetLoader(eval_cases())
    )
    monkeypatch.setattr(
        composition,
        "build_retrieval_eval",
        lambda index_path, k: FakeRetrievalEval(retrieval_report()),
    )
    monkeypatch.setattr(
        composition,
        "build_answer_eval",
        lambda index_path, k, min_score: seen.append((k, min_score))
        or FakeAnswerEval(answer_report()),
    )

    cli.main(["eval", "--k", "3", "--min-score", "0.7"])

    assert seen == [(3, 0.7)]


def test_eval_config_records_the_overridden_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 설정을 기록하지 않으면 어떤 조건의 점수인지 알 수 없다 (eval/README.md).
    use_eval_fakes(monkeypatch)
    out = tmp_path / "실험.json"

    cli.main(["eval", "--out", str(out), "--k", "3", "--min-score", "0.7"])

    config = json.loads(out.read_text(encoding="utf-8"))["config"]
    assert config["top_k"] == 3
    assert config["min_score"] == 0.7


def test_ask_accepts_a_threshold_override(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, float]] = []

    def record(path, k, min_score):  # type: ignore[no-untyped-def]
        seen.append((k, min_score))
        return FakeAnswerUsecase(Answer(body="본문", evidence=(evidence(),)))

    monkeypatch.setattr(composition, "build_answer_usecase", record)

    cli.main(["ask", "질문", "--k", "3", "--min-score", "0.7"])

    assert seen == [(3, 0.7)]


# --------------------------------------------- 색인 재료 선택 (M3-6, D-032)

def test_ingest_material_defaults_to_qa_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    builder = FakeIndexBuilder()
    monkeypatch.setattr(
        composition,
        "build_ingest",
        lambda source, material="qa", limit=None: seen.append(material)
        or FakeIngestUsecase([Chunk(id="A1#0", doc_id="A1", text="문단")]),
    )
    monkeypatch.setattr(
        composition, "build_index_builder", lambda resume_from=None: builder
    )

    cli.main(["ingest", "--aihub"])

    assert seen == ["qa"]


def test_ingest_material_can_be_switched_to_excerpt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []
    builder = FakeIndexBuilder()
    monkeypatch.setattr(
        composition,
        "build_ingest",
        lambda source, material="qa", limit=None: seen.append(material)
        or FakeIngestUsecase([Chunk(id="K-1#0", doc_id="K-1", text="문단")]),
    )
    monkeypatch.setattr(
        composition, "build_index_builder", lambda resume_from=None: builder
    )

    cli.main(["ingest", "--aihub", "--material", "excerpt"])

    assert seen == ["excerpt"]


def test_eval_config_records_the_material(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_eval_fakes(monkeypatch)
    out = tmp_path / "실험.json"

    cli.main(["eval", "--out", str(out)])

    assert "material" in json.loads(out.read_text(encoding="utf-8"))["config"]


def test_eval_config_records_the_index_file_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # material은 사람이 손으로 넣는다. 색인 이름을 같이 남겨야 어긋난 것을
    # 나중에 알아챌 수 있다 (M3-7에서 실제로 어긋났다).
    use_eval_fakes(monkeypatch)
    out = tmp_path / "실험.json"

    cli.main(["eval", "--out", str(out), "--index", "index/aihub-doc.npz"])

    config = json.loads(out.read_text(encoding="utf-8"))["config"]
    assert config["index"] == "aihub-doc.npz"


def test_eval_config_records_the_chunk_size_of_the_material(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 논문 청크는 2,048자다. 상수 하나만 읽으면 512로 잘못 기록된다.
    use_eval_fakes(monkeypatch)
    out = tmp_path / "실험.json"

    cli.main(["eval", "--out", str(out), "--material", "doc"])

    config = json.loads(out.read_text(encoding="utf-8"))["config"]
    assert config["material"] == "doc"
    assert config["chunk_size"] == 2048


def test_eval_config_keeps_the_default_chunk_size_for_qa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_eval_fakes(monkeypatch)
    out = tmp_path / "실험.json"

    cli.main(["eval", "--out", str(out)])

    assert json.loads(out.read_text(encoding="utf-8"))["config"]["chunk_size"] == 512


# ------------------------------------------------------------ serve (웹 UI)

def test_serve_builds_both_indexes_and_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[tuple[list[str], Path, int]] = []
    monkeypatch.setattr(
        composition,
        "build_answer_usecase",
        lambda path, k=5, min_score=0.35: FakeAnswerUsecase(
            Answer(body="본문", evidence=(evidence(),))
        ),
    )
    monkeypatch.setattr(
        cli.web,
        "serve",
        lambda answerers, log_path, port: started.append(
            (sorted(answerers), log_path, port)
        ),
    )

    code = cli.main(["serve", "--port", "9999"])

    assert code == 0
    assert started[0][0] == ["excerpt", "qa"]
    assert started[0][2] == 9999


def test_serve_reports_a_missing_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def missing(path: Path, k: int = 5, min_score: float = 0.35) -> FakeAnswerUsecase:
        raise FileNotFoundError(f"색인 파일이 없다: {path}")

    monkeypatch.setattr(composition, "build_answer_usecase", missing)

    code = cli.main(["serve"])

    assert code == 1
    assert "ingest" in capsys.readouterr().err
