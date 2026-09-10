"""RetrievalPolicy와 SafetyPolicy 테스트.

정책 객체는 경계에서 애매하면 버그가 된다. 임계값과 같은 값을 반드시 확인한다
(docs/guidelines/01-domain.md).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.domain.models import Chunk, Evidence
from src.domain.policy import DISCLAIMER, RetrievalPolicy, SafetyPolicy


def evidence(score: float, chunk_id: str = "d1#0") -> Evidence:
    return Evidence(chunk=Chunk(id=chunk_id, doc_id="d1", text="본문"), score=score)


# ---------------------------------------------------------- RetrievalPolicy

def test_default_values() -> None:
    policy = RetrievalPolicy()

    assert policy.k == 3
    assert policy.min_score == 0.35


def test_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        RetrievalPolicy().min_score = 0.9  # type: ignore[misc]


def test_no_evidence_is_not_sufficient() -> None:
    assert RetrievalPolicy().is_sufficient([]) is False


@pytest.mark.parametrize(
    "score,expected",
    [(0.34, False), (0.35, True), (0.36, True), (-1.0, False), (1.0, True)],
)
def test_threshold_boundary(score: float, expected: bool) -> None:
    assert RetrievalPolicy().is_sufficient([evidence(score)]) is expected


def test_sufficiency_uses_the_best_score_regardless_of_order() -> None:
    # 색인은 정렬해서 주지만, 정책이 정렬을 전제하면 다른 색인으로 바꿀 때 조용히 깨진다.
    unsorted = [evidence(0.10, "d1#0"), evidence(0.90, "d2#0")]

    assert RetrievalPolicy().is_sufficient(unsorted) is True


def test_select_keeps_only_evidence_at_or_above_threshold() -> None:
    found = [evidence(0.90, "d1#0"), evidence(0.35, "d2#0"), evidence(0.34, "d3#0")]

    selected = RetrievalPolicy().select(found)

    assert [e.chunk.id for e in selected] == ["d1#0", "d2#0"]


def test_select_preserves_input_order() -> None:
    found = [evidence(0.50, "d1#0"), evidence(0.90, "d2#0")]

    assert [e.chunk.id for e in RetrievalPolicy().select(found)] == ["d1#0", "d2#0"]


def test_select_returns_tuple_for_answer_evidence() -> None:
    # Answer.evidence는 tuple만 받는다 (MEMORY.md D-012).
    assert isinstance(RetrievalPolicy().select([evidence(0.9)]), tuple)


def test_select_on_empty_input() -> None:
    assert RetrievalPolicy().select([]) == ()


def test_sufficient_means_select_is_not_empty() -> None:
    found = [evidence(0.34)]

    policy = RetrievalPolicy()

    assert policy.is_sufficient(found) is bool(policy.select(found))


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        RetrievalPolicy(k=k)


@pytest.mark.parametrize("min_score", [-1.1, 1.1])
def test_min_score_outside_cosine_range_is_rejected(min_score: float) -> None:
    with pytest.raises(ValueError, match="min_score"):
        RetrievalPolicy(min_score=min_score)


# ------------------------------------------------------------- SafetyPolicy

SAFE_SENTENCES = [
    "자료에는 민감성 피부가 자극에 빠르게 반응한다고 기술돼 있습니다.",
    "향료 함량이 낮은 제품을 고르는 편이 낫다고 설명됩니다.",
    "증상이 지속되면 피부과 전문의와 상담하십시오.",
    "제공된 문단에는 해당 부위에 대한 설명이 다음과 같이 있습니다.",
]

DIAGNOSIS_SENTENCES = [
    "당신은 지루성 피부염입니다.",
    "이건 여드름이 확실합니다.",
    "증상을 보니 아토피로 보입니다.",
]

CURE_SENTENCES = [
    "이 제품을 바르면 낫습니다.",
    "2주 안에 완치됩니다.",
    "꾸준히 쓰면 사라집니다.",
]

PRESCRIPTION_SENTENCES = [
    "항생제를 처방해 드립니다.",
    "이 약을 복용하세요.",
]


@pytest.mark.parametrize("text", SAFE_SENTENCES)
def test_safe_sentences_pass(text: str) -> None:
    assert SafetyPolicy.default().violation(text) is None


@pytest.mark.parametrize(
    "text", DIAGNOSIS_SENTENCES + CURE_SENTENCES + PRESCRIPTION_SENTENCES
)
def test_forbidden_sentences_are_caught(text: str) -> None:
    assert SafetyPolicy.default().violation(text) is not None


def test_violation_names_the_broken_rule() -> None:
    assert SafetyPolicy.default().violation("당신은 지루성 피부염입니다.") == "진단 단정"


def test_violation_does_not_echo_the_text() -> None:
    # 위반 사유는 Refusal.detail로 흘러간다. 생성 원문이 섞이면 로그에 남는다
    # (docs/guidelines/00-common.md).
    text = "당신은 지루성 피부염입니다."

    assert text not in (SafetyPolicy.default().violation(text) or "")


def test_disclaimer_itself_is_safe() -> None:
    # 면책 문구가 걸리면 모든 Answer가 거부된다.
    assert SafetyPolicy.default().violation(DISCLAIMER) is None


def test_first_matching_rule_wins() -> None:
    text = "당신은 여드름입니다. 이 약을 복용하세요."

    assert SafetyPolicy.default().violation(text) == "진단 단정"


def test_empty_text_has_no_violation() -> None:
    assert SafetyPolicy.default().violation("") is None


def test_custom_rules_replace_the_defaults() -> None:
    policy = SafetyPolicy(rules=(("금지어", "테스트금지"),))

    assert policy.violation("테스트금지가 들어간 문장") == "금지어"
    assert policy.violation("당신은 여드름입니다.") is None


def test_safety_policy_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        SafetyPolicy.default().rules = ()  # type: ignore[misc]


def test_invalid_regex_is_rejected_at_construction() -> None:
    # 정규식 오류가 생성 시점에 드러나야 한다. 검사 시점에 터지면 답변 경로에서 죽는다.
    with pytest.raises(ValueError, match="정규식"):
        SafetyPolicy(rules=(("깨진 규칙", "[미완성"),))


# ------------------------------------ 의료 이용 판단 차단 (S-1 확장, D-034)

MEDICAL_ADVICE_SENTENCES = [
    "병원에 안 가도 됩니다.",
    "피부과에 가지 않아도 괜찮습니다.",
    "심각한 상태는 아닙니다.",
    "걱정하지 않으셔도 됩니다.",
    "2주 정도면 좋아집니다.",
    "치료가 필요하지 않습니다.",
]

MEDICAL_SAFE_SENTENCES = [
    "증상이 지속되면 피부과 전문의와 상담하십시오.",
    "자료에는 전문가 상담을 권한다고 기술돼 있습니다.",
    "자극이 심하면 병원을 방문하라고 설명됩니다.",
    "제품 사용 후 반응이 있으면 사용을 중단하라고 기술된다.",
]


@pytest.mark.parametrize("text", MEDICAL_ADVICE_SENTENCES)
def test_medical_reassurance_is_blocked(text: str) -> None:
    # "병원 안 가도 된다"는 의료 이용 판단이다. 진단만큼 위험하다.
    assert SafetyPolicy.default().violation(text) == "의료 이용 판단"


@pytest.mark.parametrize("text", MEDICAL_SAFE_SENTENCES)
def test_recommending_a_doctor_passes(text: str) -> None:
    # 병원을 권하는 문장까지 막으면 면책 문구와 정상 답변이 사라진다.
    assert SafetyPolicy.default().violation(text) is None


def test_disclaimer_still_passes_with_the_new_rule() -> None:
    # 면책 문구에 "피부과 전문의와 상담"이 들어 있다. 걸리면 모든 답이 거부된다.
    assert SafetyPolicy.default().violation(DISCLAIMER) is None
