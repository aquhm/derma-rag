"""EvaluateRetrieval / EvaluateAnswers 테스트.

검색 채점과 답변 채점을 분리한다. Recall@k는 검색만 보고, 거부율은 답변 경로
전체를 본다 (MEMORY.md D-008, eval/README.md).
"""

from __future__ import annotations

import pytest

from src.application.evaluation import EvaluateAnswers, EvaluateRetrieval, sample_cases
from src.domain.models import (
    Answer,
    Chunk,
    EvalCase,
    Evidence,
    Query,
    Refusal,
    RefusalReason,
)


def case(case_id: str, gold: tuple[str, ...] = ("d1",)) -> EvalCase:
    return EvalCase(id=case_id, question=f"{case_id} 질문", gold_doc_ids=gold)


def evidence(doc_id: str, score: float) -> Evidence:
    return Evidence(
        chunk=Chunk(id=f"{doc_id}#0", doc_id=doc_id, text="문단"), score=score
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


class FakeIndex:
    """질문 순서대로 미리 정한 검색 결과를 돌려준다."""

    def __init__(self, results: list[list[Evidence]]) -> None:
        self._results = list(results)
        self.k_values: list[int] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise AssertionError("평가 경로에서 색인에 쓰지 않는다")

    def search(self, vector: list[float], k: int) -> list[Evidence]:
        self.k_values.append(k)
        return self._results.pop(0)

    def save(self, path: object) -> None:
        raise AssertionError("평가 경로에서 색인을 저장하지 않는다")


class FakeAnswerer:
    def __init__(self, results: list[Answer | Refusal]) -> None:
        self._results = list(results)
        self.questions: list[str] = []

    def execute(self, query: Query) -> Answer | Refusal:
        self.questions.append(query.text)
        return self._results.pop(0)


def answer() -> Answer:
    return Answer(body="자료에는 이렇게 기술돼 있습니다.", evidence=(evidence("d1", 0.9),))


# -------------------------------------------------------- EvaluateRetrieval

def test_recall_counts_a_case_as_hit_when_a_gold_document_is_retrieved() -> None:
    index = FakeIndex([[evidence("d1", 0.8), evidence("d9", 0.5)]])

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute([case("Q-1")])

    assert report.hits == 1
    assert report.recall == pytest.approx(1.0)


def test_recall_counts_a_miss_when_no_gold_document_is_retrieved() -> None:
    index = FakeIndex([[evidence("d7", 0.8)]])

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute([case("Q-1")])

    assert report.hits == 0
    assert report.recall == pytest.approx(0.0)


def test_recall_is_the_ratio_over_graded_cases() -> None:
    index = FakeIndex([[evidence("d1", 0.8)], [evidence("d7", 0.8)]])

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute(
        [case("Q-1"), case("Q-2")]
    )

    assert report.recall == pytest.approx(0.5)


def test_cases_without_gold_documents_are_excluded_from_recall() -> None:
    # 근거 ID가 없으면 맞았는지 알 수 없다. 0점으로 세면 점수가 실제보다 낮아진다.
    index = FakeIndex([[evidence("d1", 0.8)], [evidence("d7", 0.8)]])

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute(
        [case("Q-1"), EvalCase(id="Q-2", question="근거 없는 질문")]
    )

    assert report.total == 2
    assert report.graded == 1
    assert report.recall == pytest.approx(1.0)


def test_any_gold_document_counts_as_a_hit() -> None:
    index = FakeIndex([[evidence("d2", 0.8)]])

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute(
        [case("Q-1", gold=("d1", "d2"))]
    )

    assert report.hits == 1


def test_k_comes_from_the_constructor() -> None:
    index = FakeIndex([[evidence("d1", 0.8)]])

    EvaluateRetrieval(FakeEmbedder(), index, k=3).execute([case("Q-1")])

    assert index.k_values == [3]


def test_questions_are_embedded_in_one_batch() -> None:
    embedder = FakeEmbedder()
    index = FakeIndex([[evidence("d1", 0.8)], [evidence("d1", 0.8)]])

    EvaluateRetrieval(embedder, index, k=5).execute([case("Q-1"), case("Q-2")])

    assert embedder.calls == [["Q-1 질문", "Q-2 질문"]]


def test_report_keeps_the_best_score_per_case() -> None:
    index = FakeIndex([[evidence("d1", 0.42), evidence("d2", 0.31)]])

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute([case("Q-1")])

    assert report.results[0].best_score == pytest.approx(0.42)
    assert report.mean_best_score == pytest.approx(0.42)


def test_empty_search_result_gives_zero_best_score() -> None:
    report = EvaluateRetrieval(FakeEmbedder(), FakeIndex([[]]), k=5).execute(
        [case("Q-1")]
    )

    assert report.results[0].best_score == pytest.approx(0.0)
    assert report.hits == 0


def test_retrieval_progress_is_reported() -> None:
    index = FakeIndex([[evidence("d1", 0.8)], [evidence("d1", 0.8)]])
    seen: list[tuple[int, int]] = []

    EvaluateRetrieval(FakeEmbedder(), index, k=5).execute(
        [case("Q-1"), case("Q-2")], on_progress=lambda done, total: seen.append((done, total))
    )

    assert seen == [(1, 2), (2, 2)]


def test_retrieval_report_serializes_for_eval_files() -> None:
    index = FakeIndex([[evidence("d1", 0.8)]])

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute([case("Q-1")])
    payload = report.to_dict()

    assert payload["recall_at_5"] == pytest.approx(1.0)
    assert payload["sample_count"] == 1
    assert payload["graded_count"] == 1


def test_no_cases_is_an_error() -> None:
    with pytest.raises(ValueError, match="평가 케이스가 없다"):
        EvaluateRetrieval(FakeEmbedder(), FakeIndex([]), k=5).execute([])


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        EvaluateRetrieval(FakeEmbedder(), FakeIndex([]), k=k)


def test_report_never_contains_question_text() -> None:
    # 질문은 개인 건강 정보일 수 있다 (SAFETY.md S-7). 보고서에는 ID만 남는다.
    index = FakeIndex([[evidence("d1", 0.8)]])
    question = "민감성 피부가 세안 후 당길 때 무엇을 주의해야 하나요?"

    report = EvaluateRetrieval(FakeEmbedder(), index, k=5).execute(
        [EvalCase(id="Q-1", question=question, gold_doc_ids=("d1",))]
    )

    assert question not in str(report.to_dict())
    assert question not in str(report.results)


# ---------------------------------------------------------- EvaluateAnswers

def test_answer_report_counts_answers_and_refusals() -> None:
    answerer = FakeAnswerer(
        [
            answer(),
            Refusal(reason=RefusalReason.NO_EVIDENCE, detail="검색 결과가 없다"),
            Refusal(reason=RefusalReason.SAFETY_VIOLATION, detail="진단 단정"),
        ]
    )

    report = EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2"), case("Q-3")])

    assert report.total == 3
    assert report.answered == 1
    assert report.refused == 2
    assert report.refusal_rate == pytest.approx(2 / 3)


def test_answer_report_splits_refusals_by_reason() -> None:
    answerer = FakeAnswerer(
        [
            Refusal(reason=RefusalReason.NO_EVIDENCE, detail=""),
            Refusal(reason=RefusalReason.SAFETY_VIOLATION, detail="진단 단정"),
        ]
    )

    report = EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2")])

    assert report.by_reason["근거 없음"] == 1
    assert report.by_reason["안전 규칙 위반"] == 1


def test_answer_report_serializes_for_eval_files() -> None:
    answerer = FakeAnswerer([answer(), Refusal(reason=RefusalReason.NO_EVIDENCE)])

    payload = EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2")]).to_dict()

    assert payload["refusal_rate"] == pytest.approx(0.5)
    assert payload["sample_count"] == 2


def test_every_case_reaches_the_answer_usecase() -> None:
    answerer = FakeAnswerer([answer(), answer()])

    EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2")])

    assert answerer.questions == ["Q-1 질문", "Q-2 질문"]


def test_answer_progress_is_reported() -> None:
    answerer = FakeAnswerer([answer(), answer()])
    seen: list[tuple[int, int]] = []

    EvaluateAnswers(answerer).execute(
        [case("Q-1"), case("Q-2")],
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(1, 2), (2, 2)]


def test_answer_evaluation_rejects_an_empty_case_list() -> None:
    with pytest.raises(ValueError, match="평가 케이스가 없다"):
        EvaluateAnswers(FakeAnswerer([])).execute([])


# -------------------------------------------------- 답변 정확도 (M2-7, D-026)

def graded_case(case_id: str, expected: str) -> EvalCase:
    return EvalCase(
        id=case_id, question=f"{case_id} 질문", expected=expected, gold_doc_ids=("d1",)
    )


def answer_saying(body: str) -> Answer:
    return Answer(body=body, evidence=(evidence("d1", 0.9),))


def test_accuracy_is_the_mean_bigram_f1_of_answered_cases() -> None:
    answerer = FakeAnswerer([answer_saying("향료를 피한다"), answer_saying("향료를 피한다")])

    report = EvaluateAnswers(answerer).execute(
        [graded_case("Q-1", "향료를 피한다"), graded_case("Q-2", "향료를 피한다")]
    )

    assert report.accuracy == pytest.approx(1.0)
    assert report.scored == 2


def test_wrong_answer_lowers_the_accuracy() -> None:
    answerer = FakeAnswerer([answer_saying("향료를 피한다"), answer_saying("자동차 엔진 오일")])

    report = EvaluateAnswers(answerer).execute(
        [graded_case("Q-1", "향료를 피한다"), graded_case("Q-2", "향료를 피한다")]
    )

    assert report.accuracy == pytest.approx(0.5)


def test_refused_cases_are_excluded_from_accuracy() -> None:
    # 거부는 틀린 답이 아니라 답하지 않은 것이다. 0점으로 섞으면 거부율과 정확도가
    # 같은 것을 두 번 세게 된다 (MEMORY.md D-008, D-026).
    answerer = FakeAnswerer(
        [answer_saying("향료를 피한다"), Refusal(reason=RefusalReason.NO_EVIDENCE)]
    )

    report = EvaluateAnswers(answerer).execute(
        [graded_case("Q-1", "향료를 피한다"), graded_case("Q-2", "향료를 피한다")]
    )

    assert report.accuracy == pytest.approx(1.0)
    assert report.scored == 1
    assert report.refusal_rate == pytest.approx(0.5)


def test_cases_without_expected_text_are_not_scored() -> None:
    answerer = FakeAnswerer([answer_saying("아무 말")])

    report = EvaluateAnswers(answerer).execute([case("Q-1")])

    assert report.scored == 0
    assert report.accuracy == 0.0


def test_accuracy_appears_in_the_result_file() -> None:
    answerer = FakeAnswerer([answer_saying("향료를 피한다")])

    payload = EvaluateAnswers(answerer).execute(
        [graded_case("Q-1", "향료를 피한다")]
    ).to_dict()

    assert payload["answer_accuracy"] == pytest.approx(1.0)
    assert payload["scored_count"] == 1


def test_similarity_is_recorded_when_an_embedder_is_given() -> None:
    class ConstantEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    answerer = FakeAnswerer([answer_saying("표현이 전혀 다른 문장")])

    report = EvaluateAnswers(answerer, embedder=ConstantEmbedder()).execute(
        [graded_case("Q-1", "향료를 피한다")]
    )

    assert report.similarity == pytest.approx(1.0)
    assert report.to_dict()["answer_similarity"] == pytest.approx(1.0)


def test_similarity_is_absent_without_an_embedder() -> None:
    answerer = FakeAnswerer([answer_saying("향료를 피한다")])

    payload = EvaluateAnswers(answerer).execute(
        [graded_case("Q-1", "향료를 피한다")]
    ).to_dict()

    assert "answer_similarity" not in payload


# ------------------------------------------------------------ 표본 추출

def test_sampling_returns_the_requested_number() -> None:
    cases = [case(f"Q-{i}") for i in range(50)]

    assert len(sample_cases(cases, size=10, seed=42)) == 10


def test_sampling_is_reproducible_with_the_same_seed() -> None:
    # 실험마다 다른 표본을 쓰면 점수 차이가 변경 때문인지 표본 때문인지 알 수 없다.
    cases = [case(f"Q-{i}") for i in range(50)]

    first = [c.id for c in sample_cases(cases, size=10, seed=42)]
    second = [c.id for c in sample_cases(cases, size=10, seed=42)]

    assert first == second


def test_a_different_seed_gives_a_different_sample() -> None:
    cases = [case(f"Q-{i}") for i in range(50)]

    first = [c.id for c in sample_cases(cases, size=10, seed=42)]
    second = [c.id for c in sample_cases(cases, size=10, seed=7)]

    assert first != second


def test_sampling_keeps_the_original_order() -> None:
    cases = [case(f"Q-{i}") for i in range(50)]

    sampled = sample_cases(cases, size=10, seed=42)

    assert [c.id for c in sampled] == [c.id for c in cases if c in sampled]


def test_sampling_more_than_available_returns_everything() -> None:
    cases = [case(f"Q-{i}") for i in range(5)]

    assert len(sample_cases(cases, size=10, seed=42)) == 5


def test_size_zero_means_everything() -> None:
    cases = [case(f"Q-{i}") for i in range(5)]

    assert len(sample_cases(cases, size=0, seed=42)) == 5


# ------------------------------------------------ 생성 실패 견디기 (E-003)

class FlakyAnswerer:
    """케이스마다 정해진 결과나 예외를 돌려준다."""

    def __init__(self, results: list[object]) -> None:
        self._results = list(results)
        self.calls = 0

    def execute(self, query: Query):  # type: ignore[no-untyped-def]
        self.calls += 1
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_one_generation_failure_does_not_stop_the_run() -> None:
    # 200건 평가가 20분 걸린다. 한 건이 타임아웃 났다고 전부 잃으면 안 된다.
    answerer = FlakyAnswerer([answer(), RuntimeError("read timeout"), answer()])

    report = EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2"), case("Q-3")])

    assert answerer.calls == 3
    assert report.total == 3
    assert report.failed == 1


def test_failed_cases_are_not_counted_as_answers_or_refusals() -> None:
    answerer = FlakyAnswerer([answer(), RuntimeError("read timeout")])

    report = EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2")])

    assert report.answered == 1
    assert report.refused == 0
    # 거부율은 끝까지 돈 케이스(답변 + 거부) 기준이다. 실패는 분모에서 빠진다.
    assert report.refusal_rate == pytest.approx(0.0)


def test_failure_count_appears_in_the_result_file() -> None:
    answerer = FlakyAnswerer([answer(), RuntimeError("read timeout")])

    payload = EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2")]).to_dict()

    assert payload["failed_count"] == 1


def test_failure_messages_do_not_carry_the_question() -> None:
    question = "민감성 피부가 세안 후 당길 때 무엇을 주의해야 하나요?"
    answerer = FlakyAnswerer([RuntimeError("read timeout")])

    report = EvaluateAnswers(answerer).execute(
        [EvalCase(id="Q-1", question=question, gold_doc_ids=("d1",))]
    )

    assert all(question not in message for message in report.failures)


def test_every_case_failing_is_reported_not_raised() -> None:
    answerer = FlakyAnswerer([RuntimeError("a"), RuntimeError("b")])

    report = EvaluateAnswers(answerer).execute([case("Q-1"), case("Q-2")])

    assert report.failed == 2
    assert report.accuracy == 0.0


# --------------------------------------------- 케이스별 결과 기록 (O-5)

def test_report_keeps_one_row_per_case() -> None:
    answerer = FlakyAnswerer(
        [answer_saying("향료를 피한다"), Refusal(reason=RefusalReason.NO_EVIDENCE), RuntimeError("timeout")]
    )

    report = EvaluateAnswers(answerer).execute(
        [graded_case("Q-1", "향료를 피한다"), case("Q-2"), case("Q-3")]
    )

    assert [c.case_id for c in report.cases] == ["Q-1", "Q-2", "Q-3"]
    assert [c.outcome for c in report.cases] == ["answered", "refused", "failed"]


def test_case_row_keeps_the_accuracy_score() -> None:
    answerer = FlakyAnswerer([answer_saying("향료를 피한다")])

    report = EvaluateAnswers(answerer).execute([graded_case("Q-1", "향료를 피한다")])

    assert report.cases[0].accuracy == pytest.approx(1.0)


def test_case_row_without_expected_text_has_no_accuracy() -> None:
    answerer = FlakyAnswerer([answer_saying("아무 말")])

    report = EvaluateAnswers(answerer).execute([case("Q-1")])

    assert report.cases[0].accuracy is None


def test_case_row_keeps_the_refusal_reason() -> None:
    answerer = FlakyAnswerer([Refusal(reason=RefusalReason.SAFETY_VIOLATION, detail="진단 단정")])

    report = EvaluateAnswers(answerer).execute([case("Q-1")])

    assert report.cases[0].detail == "안전 규칙 위반"


def test_case_row_keeps_the_failure_type() -> None:
    answerer = FlakyAnswerer([RuntimeError("read timeout")])

    report = EvaluateAnswers(answerer).execute([case("Q-1")])

    assert report.cases[0].detail == "RuntimeError"


def test_case_rows_never_contain_question_or_answer_text() -> None:
    # 질문은 개인 건강 정보일 수 있다 (SAFETY.md S-7). 답변에도 원본 문장이 섞인다.
    question = "민감성 피부가 세안 후 당길 때 무엇을 주의해야 하나요?"
    body = "자료에는 향료를 피하라고 기술돼 있습니다."
    answerer = FlakyAnswerer([Answer(body=body, evidence=(evidence("d1", 0.9),))])

    report = EvaluateAnswers(answerer).execute(
        [EvalCase(id="Q-1", question=question, expected="기대", gold_doc_ids=("d1",))]
    )

    row = str(report.cases[0])
    assert question not in row
    assert body not in row


def test_case_rows_carry_similarity_when_measured() -> None:
    class ConstantEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    answerer = FlakyAnswerer([answer_saying("아무 말")])

    report = EvaluateAnswers(answerer, embedder=ConstantEmbedder()).execute(
        [graded_case("Q-1", "기대 답변")]
    )

    assert report.cases[0].similarity == pytest.approx(1.0)
