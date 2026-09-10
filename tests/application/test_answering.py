"""AnswerQuestion 테스트.

포트는 가짜 클래스로 대체한다. unittest.mock을 쓰지 않는 이유는 포트 시그니처가
바뀌어도 목은 조용히 통과하기 때문이다 (docs/guidelines/05-testing.md).

이 파일에 필수 테스트 3종이 전부 들어 있다.

1. 근거가 부족하면 생성기를 호출하지 않는다 (SAFETY.md S-2)
2. 진단 표현이 걸러진다 (SAFETY.md S-1, S-4)
3. 면책 문구가 항상 붙는다 (SAFETY.md S-5)
"""

from __future__ import annotations

import pytest

from src.application.answering import NO_ANSWER_TOKEN, AnswerQuestion
from src.domain.models import (
    Answer,
    Chunk,
    Evidence,
    Query,
    Refusal,
    RefusalReason,
    SkinType,
)
from src.domain.policy import DISCLAIMER, RetrievalPolicy, SafetyPolicy


def evidence(score: float, chunk_id: str = "d1#0", text: str = "본문") -> Evidence:
    return Evidence(
        chunk=Chunk(id=chunk_id, doc_id=chunk_id.split("#")[0], text=text),
        score=score,
    )


class FakeEmbedder:
    """application.ports.Embedder 대역."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self._vector = [1.0, 0.0] if vector is None else vector
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector for _ in texts]


class EmptyEmbedder:
    """벡터를 하나도 돌려주지 않는 고장 난 어댑터."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []


class FakeIndex:
    """application.ports.VectorIndex 대역. 검색 인자를 기록한다."""

    def __init__(self, found: list[Evidence] | None = None) -> None:
        self._found = found if found is not None else []
        self.searches: list[tuple[list[float], int]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        raise AssertionError("답변 경로에서 색인에 쓰지 않는다")

    def search(self, vector: list[float], k: int) -> list[Evidence]:
        self.searches.append((vector, k))
        return self._found[:k]

    def save(self, path: object) -> None:
        raise AssertionError("답변 경로에서 색인을 저장하지 않는다")


class FakeGenerator:
    """application.ports.Generator 대역. 호출 여부를 센다."""

    def __init__(self, response: str = "자료에는 다음과 같이 기술돼 있습니다.") -> None:
        self._response = response
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


def build(
    found: list[Evidence] | None = None,
    generator: FakeGenerator | None = None,
    embedder: FakeEmbedder | None = None,
    index: FakeIndex | None = None,
    retrieval_policy: RetrievalPolicy | None = None,
) -> AnswerQuestion:
    return AnswerQuestion(
        embedder=embedder if embedder is not None else FakeEmbedder(),
        index=index if index is not None else FakeIndex(found),
        generator=generator if generator is not None else FakeGenerator(),
        retrieval_policy=(
            retrieval_policy if retrieval_policy is not None else RetrievalPolicy()
        ),
        safety_policy=SafetyPolicy.default(),
    )


# --------------------------------------------------- 필수 테스트 1 · 근거 없음

def test_no_evidence_skips_generation() -> None:
    generator = FakeGenerator()

    result = build(found=[], generator=generator).execute(Query("무관한 질문"))

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NO_EVIDENCE
    assert generator.calls == []


def test_low_score_evidence_skips_generation() -> None:
    generator = FakeGenerator()

    result = build(found=[evidence(0.34)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NO_EVIDENCE
    assert generator.calls == []


def test_threshold_score_is_answered() -> None:
    result = build(found=[evidence(0.35)]).execute(Query("질문"))

    assert isinstance(result, Answer)


# ------------------------------------------------- 필수 테스트 2 · 진단 표현

def test_diagnosis_in_generated_text_becomes_refusal() -> None:
    generator = FakeGenerator("당신은 지루성 피부염입니다.")

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.SAFETY_VIOLATION
    assert result.detail == "진단 단정"


def test_refusal_detail_does_not_carry_generated_text() -> None:
    unsafe = "이 제품을 바르면 낫습니다."
    generator = FakeGenerator(unsafe)

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert unsafe not in result.detail


# ------------------------------------------------- 필수 테스트 3 · 면책 문구

def test_answer_always_carries_the_disclaimer() -> None:
    result = build(found=[evidence(0.9)]).execute(Query("질문"))

    assert isinstance(result, Answer)
    assert DISCLAIMER in result.text


# ------------------------------------------------------------------ 검색 경로

def test_question_is_embedded_once() -> None:
    embedder = FakeEmbedder()

    build(found=[evidence(0.9)], embedder=embedder).execute(Query("민감성 피부 질문"))

    assert embedder.calls == [["민감성 피부 질문"]]


def test_search_uses_k_from_policy() -> None:
    index = FakeIndex([evidence(0.9, f"d{i}#0") for i in range(10)])

    build(index=index, retrieval_policy=RetrievalPolicy(k=3)).execute(Query("질문"))

    assert index.searches[0][1] == 3


def test_answer_carries_only_evidence_above_threshold() -> None:
    found = [evidence(0.9, "d1#0"), evidence(0.5, "d2#0"), evidence(0.2, "d3#0")]

    result = build(found=found).execute(Query("질문"))

    assert isinstance(result, Answer)
    assert [e.chunk.id for e in result.evidence] == ["d1#0", "d2#0"]


def test_embedder_returning_nothing_is_an_error() -> None:
    usecase = AnswerQuestion(
        embedder=EmptyEmbedder(),
        index=FakeIndex([evidence(0.9)]),
        generator=FakeGenerator(),
        retrieval_policy=RetrievalPolicy(),
        safety_policy=SafetyPolicy.default(),
    )

    with pytest.raises(ValueError, match="질문 벡터"):
        usecase.execute(Query("질문"))


def test_execute_rejects_a_plain_string() -> None:
    # Query는 __post_init__에서 빈 문자열을 막는다. 문자열을 그대로 받으면 그 검사를
    # 건너뛴다.
    with pytest.raises(TypeError, match="Query"):
        build(found=[evidence(0.9)]).execute("질문")  # type: ignore[arg-type]


# -------------------------------------------------------------- 프롬프트 조립

def test_prompt_contains_the_question() -> None:
    generator = FakeGenerator()

    build(found=[evidence(0.9)], generator=generator).execute(Query("모공이 넓어 보입니다"))

    assert "모공이 넓어 보입니다" in generator.calls[0]


def test_prompt_contains_selected_evidence_only() -> None:
    found = [
        evidence(0.9, "d1#0", "선택된 문단"),
        evidence(0.1, "d2#0", "버려질 문단"),
    ]
    generator = FakeGenerator()

    build(found=found, generator=generator).execute(Query("질문"))

    prompt = generator.calls[0]
    assert "선택된 문단" in prompt
    assert "버려질 문단" not in prompt


def test_prompt_forbids_outside_knowledge() -> None:
    # SAFETY.md S-3. 프롬프트만으로 충분하지는 않지만 있어야 한다.
    generator = FakeGenerator()

    build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    prompt = generator.calls[0]
    assert "제공된 문단" in prompt
    assert NO_ANSWER_TOKEN in prompt


def test_prompt_labels_evidence_with_chunk_metadata() -> None:
    found = [
        Evidence(
            chunk=Chunk(
                id="d1#0",
                doc_id="d1",
                text="이마는 피지 분비가 많은 구간으로 기술된다.",
                skin_type=SkinType.INFLAMMATORY,
                area="이마",
            ),
            score=0.9,
        )
    ]
    generator = FakeGenerator()

    build(found=found, generator=generator).execute(Query("질문"))

    prompt = generator.calls[0]
    assert "d1#0" in prompt
    assert "염증성" in prompt
    assert "이마" in prompt


# ------------------------------------------------------------ 생성 결과 처리

def test_sentinel_answer_becomes_refusal() -> None:
    # 모델이 지시대로 "자료에 없다"고 답하면 그것은 답이 아니라 거부다.
    generator = FakeGenerator(NO_ANSWER_TOKEN)

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NO_ANSWER_FROM_MODEL


def test_blank_generation_becomes_refusal() -> None:
    generator = FakeGenerator("   \n  ")

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NO_ANSWER_FROM_MODEL


def test_answer_body_is_the_generated_text() -> None:
    generator = FakeGenerator("자료에는 세정 후 보습이 권장된다고 기술돼 있습니다.")

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Answer)
    assert result.body == "자료에는 세정 후 보습이 권장된다고 기술돼 있습니다."


# ------------------------------------------------- 자료 없음 신호 (실측 반영)

def test_no_answer_token_becomes_refusal() -> None:
    # 모델에게 시키는 것은 고정 토큰이다. 한국어 문장으로 시키면 표현이 매번 달라져
    # 문자열 비교가 실패한다 (2026-09-08 실측, ERRORS.md E-002).
    generator = FakeGenerator("NO_ANSWER")

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NO_ANSWER_FROM_MODEL


def test_no_answer_token_inside_a_sentence_still_refuses() -> None:
    generator = FakeGenerator("NO_ANSWER 입니다.")

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)


@pytest.mark.parametrize(
    "text",
    [
        "제공된 자료에 자동차 엔진 오일 교환에 관한 내용이 없습니다.",
        "제공된 문단에는 해당 내용이 없습니다.",
        "죄송합니다. 제공된 자료에서 관련 내용을 찾을 수 없습니다.",
        "주어진 자료에는 그 내용이 없다.",
    ],
)
def test_paraphrased_no_answer_becomes_refusal(text: str) -> None:
    generator = FakeGenerator(text)

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert result.reason is RefusalReason.NO_ANSWER_FROM_MODEL


@pytest.mark.parametrize(
    "text",
    [
        "자료에는 향료가 없는 제품을 고르라고 기술돼 있습니다.",
        "자극이 적은 제품에는 알코올이 없습니다.",
        "해당 부위에는 각질이 없다고 기술돼 있습니다.",
    ],
)
def test_normal_answers_containing_the_word_are_not_refused(text: str) -> None:
    # "없습니다"만 보고 거부하면 정상 답변이 사라진다.
    generator = FakeGenerator(text)

    result = build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert isinstance(result, Answer)


def test_prompt_tells_the_model_which_token_to_use() -> None:
    generator = FakeGenerator()

    build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert NO_ANSWER_TOKEN in generator.calls[0]


def test_refusal_detail_shows_the_best_score_and_the_threshold() -> None:
    # CLI가 왜 거부됐는지 보여줘야 한다 (docs/guidelines/04-interface.md).
    result = build(found=[evidence(0.21), evidence(0.19, "d2#0")]).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert "0.21" in result.detail
    assert "0.35" in result.detail


def test_refusal_detail_without_any_search_hit() -> None:
    result = build(found=[]).execute(Query("질문"))

    assert isinstance(result, Refusal)
    assert "검색 결과가 없다" in result.detail


def test_prompt_asks_for_a_bounded_length() -> None:
    # M3-4. 길이 제약이 빠지면 답변이 길어지고 F1 정밀도가 떨어진다.
    generator = FakeGenerator()

    build(found=[evidence(0.9)], generator=generator).execute(Query("질문"))

    assert "400자" in generator.calls[0]
