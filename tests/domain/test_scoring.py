"""답변 정확도 채점 함수 테스트 (M2-7, MEMORY.md D-026).

음절 2-gram F1이다. 한국어는 조사가 붙어 어절 단위 비교가 쉽게 어긋난다.
음절 단위로 보면 "피한다"와 "피하고"가 부분 점수를 받는다.
"""

from __future__ import annotations

import pytest

from src.domain.scoring import bigram_f1, syllable_bigrams


# ------------------------------------------------------------ 2-gram 추출

def test_bigrams_of_a_short_word() -> None:
    assert syllable_bigrams("피부") == ["피부"]


def test_bigrams_slide_by_one_syllable() -> None:
    assert syllable_bigrams("민감성") == ["민감", "감성"]


def test_whitespace_and_punctuation_are_dropped() -> None:
    # 띄어쓰기와 문장부호가 다르다고 점수가 달라지면 안 된다.
    assert syllable_bigrams("피부, 관리!") == syllable_bigrams("피부관리")


def test_a_single_character_still_gives_one_token() -> None:
    # 2-gram을 만들 수 없는 짧은 입력은 그 글자 자체를 쓴다.
    assert syllable_bigrams("볼") == ["볼"]


def test_empty_text_has_no_bigrams() -> None:
    assert syllable_bigrams("   ") == []


def test_letters_are_lowercased() -> None:
    assert syllable_bigrams("AB") == syllable_bigrams("ab")


# ---------------------------------------------------------------- F1 계산

def test_identical_text_scores_one() -> None:
    assert bigram_f1("향료를 피한다", "향료를 피한다") == pytest.approx(1.0)


def test_completely_different_text_scores_zero() -> None:
    assert bigram_f1("향료를 피한다", "자동차 엔진 오일") == pytest.approx(0.0)


def test_partial_overlap_scores_between_zero_and_one() -> None:
    score = bigram_f1(
        "향료와 알코올을 피하라고 기술된다",
        "향료와 알코올 함량이 높은 제품을 피한다",
    )

    assert 0.0 < score < 1.0


def test_score_is_symmetric() -> None:
    a, b = "향료를 피한다", "향료 함량이 높은 제품을 피한다"

    assert bigram_f1(a, b) == pytest.approx(bigram_f1(b, a))


def test_padding_lowers_the_score() -> None:
    # 정답을 포함하되 장황한 답은 정밀도가 떨어져 점수가 낮아야 한다. 재현율만
    # 쓰면 길게 쓸수록 유리해진다.
    exact = bigram_f1("향료를 피한다", "향료를 피한다")
    padded = bigram_f1("향료를 피한다", "향료를 피한다 " + "그리고 여러 가지 설명 " * 10)

    assert padded < exact


def test_repeated_words_do_not_inflate_the_score() -> None:
    # 같은 2-gram을 반복해 재현율을 끌어올리는 것을 막는다.
    honest = bigram_f1("향료를 피한다", "향료를 피한다")
    spammed = bigram_f1("향료를 피한다", "향료 향료 향료 향료 향료")

    assert spammed < honest


def test_empty_generated_text_scores_zero() -> None:
    assert bigram_f1("향료를 피한다", "") == pytest.approx(0.0)


def test_empty_expected_text_scores_zero() -> None:
    # 기대 답변이 없으면 채점할 수 없다. 1.0을 주면 채점 안 된 케이스가 만점이 된다.
    assert bigram_f1("", "향료를 피한다") == pytest.approx(0.0)


def test_both_empty_scores_zero() -> None:
    assert bigram_f1("", "") == pytest.approx(0.0)


def test_score_stays_within_range() -> None:
    score = bigram_f1("민감성 피부는 자극에 반응한다", "민감성 피부 자극 반응")

    assert 0.0 <= score <= 1.0
