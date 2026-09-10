"""domain 값 객체 단위 테스트.

목이 없다. domain은 외부에 의존하지 않으므로 Ollama가 꺼져 있어도 돈다.
테스트 문서는 전부 코드 안에 둔 가짜 문장이다. data/ 아래 원본을 쓰지 않는다
(MEMORY.md D-006, docs/guidelines/05-testing.md).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.domain.models import (
    Answer,
    Chunk,
    Document,
    EvalCase,
    Evidence,
    Query,
    Refusal,
    RefusalReason,
    Result,
    SkinType,
)
from src.domain.policy import DISCLAIMER


def make_chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id="d1",
        text="가짜 문단이다. 실제 데이터가 아니다.",
        skin_type=SkinType.SENSITIVE,
        area="볼",
    )


# --- Document ---------------------------------------------------------------


def test_document_holds_metadata() -> None:
    doc = Document(id="d1", text="본문", skin_type=SkinType.INFLAMMATORY, area="이마")

    assert doc.skin_type is SkinType.INFLAMMATORY
    assert doc.area == "이마"


def test_document_metadata_is_optional() -> None:
    doc = Document(id="d1", text="본문")

    assert doc.skin_type is None
    assert doc.area is None


@pytest.mark.parametrize("doc_id,text", [("", "본문"), ("   ", "본문"), ("d1", "  ")])
def test_document_rejects_empty_field(doc_id: str, text: str) -> None:
    with pytest.raises(ValueError):
        Document(id=doc_id, text=text)


# --- Chunk ------------------------------------------------------------------


def test_chunk_knows_its_document() -> None:
    chunk = make_chunk()

    assert chunk.doc_id == "d1"


@pytest.mark.parametrize(
    "chunk_id,doc_id,text",
    [("", "d1", "본문"), ("c1", "", "본문"), ("c1", "d1", "")],
)
def test_chunk_rejects_empty_field(chunk_id: str, doc_id: str, text: str) -> None:
    with pytest.raises(ValueError):
        Chunk(id=chunk_id, doc_id=doc_id, text=text)


def test_chunk_is_immutable() -> None:
    chunk = make_chunk()

    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "바꾼다"  # type: ignore[misc]


# --- Query ------------------------------------------------------------------


def test_query_rejects_blank() -> None:
    with pytest.raises(ValueError):
        Query(text="   ")


# --- Evidence ---------------------------------------------------------------


@pytest.mark.parametrize("score", [-1.0, -0.5, 0.0, 0.35, 1.0])
def test_evidence_accepts_valid_score(score: float) -> None:
    assert Evidence(chunk=make_chunk(), score=score).score == score


@pytest.mark.parametrize("score", [-1.0001, 1.0001, 2.0, -3.0])
def test_evidence_rejects_score_out_of_range(score: float) -> None:
    with pytest.raises(ValueError):
        Evidence(chunk=make_chunk(), score=score)


# --- Answer -----------------------------------------------------------------


def make_answer(body: str = "데이터셋에는 다음과 같이 기술돼 있습니다.") -> Answer:
    return Answer(body=body, evidence=(Evidence(chunk=make_chunk(), score=0.7),))


def test_answer_always_appends_disclaimer() -> None:
    """SAFETY.md S-5. 면책 문구는 조건부로 붙지 않는다."""
    assert DISCLAIMER in make_answer().text


def test_answer_body_does_not_carry_disclaimer() -> None:
    """면책은 body에 섞여 들어가지 않는다. 붙이는 곳은 text 한 군데뿐이다."""
    assert DISCLAIMER not in make_answer().body


def test_answer_without_evidence_cannot_exist() -> None:
    """MEMORY.md D-007. 근거 없는 답은 값으로도 만들 수 없다."""
    with pytest.raises(ValueError):
        Answer(body="본문", evidence=())


def test_answer_rejects_list_evidence() -> None:
    """list를 받으면 frozen이 무의미해진다. append로 내용이 바뀌기 때문이다."""
    with pytest.raises(TypeError):
        Answer(body="본문", evidence=[Evidence(chunk=make_chunk(), score=0.7)])  # type: ignore[arg-type]


@pytest.mark.parametrize("body", ["", "   ", "\n"])
def test_answer_rejects_empty_body(body: str) -> None:
    with pytest.raises(ValueError):
        Answer(body=body, evidence=(Evidence(chunk=make_chunk(), score=0.7),))


# --- Refusal ----------------------------------------------------------------


def test_refusal_carries_reason() -> None:
    refusal = Refusal(reason=RefusalReason.NO_EVIDENCE)

    assert refusal.reason is RefusalReason.NO_EVIDENCE
    assert refusal.detail == ""


def test_refusal_rejects_raw_string_reason() -> None:
    """사유를 문자열로 넘기면 사유별 분기가 조용히 어긋난다."""
    with pytest.raises(TypeError):
        Refusal(reason="근거 없음")  # type: ignore[arg-type]


def test_refusal_reasons_cover_every_failure_point() -> None:
    """실패 지점 셋에 각각 대응하는 사유가 있다.

    검색(SAFETY.md S-2), 생성, 생성 후 검사(S-4)다. 앞의 둘을 한 값으로 묶으면
    보고서를 보고 어디를 고칠지 알 수 없다 (MEMORY.md O-11).
    """
    assert {r.name for r in RefusalReason} == {
        "NO_EVIDENCE",
        "NO_ANSWER_FROM_MODEL",
        "SAFETY_VIOLATION",
    }


# --- Result -----------------------------------------------------------------


def test_result_forces_branching() -> None:
    """Result를 받은 쪽은 두 경우를 모두 처리해야 한다."""
    values: list[Result] = [make_answer(), Refusal(reason=RefusalReason.NO_EVIDENCE)]

    rendered = [v.text if isinstance(v, Answer) else v.reason.value for v in values]

    assert DISCLAIMER in rendered[0]
    assert rendered[1] == "근거 없음"


# --- SkinType ---------------------------------------------------------------


def test_skin_type_has_four_categories() -> None:
    assert {s.value for s in SkinType} == {"민감성", "염증성", "색소문제", "조직변화"}


# ------------------------------------------------------------------ EvalCase

def test_eval_case_holds_question_and_gold_documents() -> None:
    case = EvalCase(id="Q-1", question="민감성 피부 질문", gold_doc_ids=("sample-001",))

    assert case.id == "Q-1"
    assert case.gold_doc_ids == ("sample-001",)


def test_eval_case_defaults_are_empty() -> None:
    case = EvalCase(id="Q-1", question="질문")

    assert case.expected == ""
    assert case.gold_doc_ids == ()


def test_eval_case_rejects_blank_id() -> None:
    with pytest.raises(ValueError, match="EvalCase.id"):
        EvalCase(id="  ", question="질문")


def test_eval_case_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="질문"):
        EvalCase(id="Q-1", question="   ")


def test_eval_case_rejects_a_list_of_gold_ids() -> None:
    # frozen=True인데 필드가 list면 내용이 바뀐다. Answer.evidence와 같은 이유다.
    with pytest.raises(TypeError, match="tuple"):
        EvalCase(id="Q-1", question="질문", gold_doc_ids=["sample-001"])  # type: ignore[arg-type]


def test_eval_case_is_gradable_only_with_gold_documents() -> None:
    # QA쌍이 근거 문단 ID를 갖지 않을 수 있다 (M2-1 미확인). 그 경우 Recall을
    # 계산할 수 없으므로 채점 대상에서 빼야 한다.
    assert EvalCase(id="Q-1", question="질문", gold_doc_ids=("d1",)).is_gradable is True
    assert EvalCase(id="Q-2", question="질문").is_gradable is False


# --------------------------------------------------- 세부 피부 유형 (D-028)

def test_document_carries_a_detail_type() -> None:
    doc = Document(
        id="A-1",
        text="본문",
        skin_type=SkinType.SENSITIVE,
        skin_detail="아토피 피부",
        area="볼",
    )

    assert doc.skin_detail == "아토피 피부"


def test_document_detail_type_defaults_to_none() -> None:
    assert Document(id="A-1", text="본문").skin_detail is None


def test_chunk_carries_a_detail_type() -> None:
    chunk = Chunk(
        id="A-1#0", doc_id="A-1", text="본문", skin_detail="지루성 피부염"
    )

    assert chunk.skin_detail == "지루성 피부염"


def test_blank_detail_type_becomes_none() -> None:
    # 실데이터에 빈 문자열이 들어 있다. 빈 문자열을 그대로 두면 "값이 있다"로
    # 취급되어 필터링(M3-2)이 어긋난다.
    assert Document(id="A-1", text="본문", skin_detail="   ").skin_detail is None
    assert Chunk(id="A-1#0", doc_id="A-1", text="본문", skin_detail="").skin_detail is None
