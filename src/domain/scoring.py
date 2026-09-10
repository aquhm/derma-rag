"""답변 정확도 채점 (M2-7, MEMORY.md D-026).

기대 답변과 생성 답변을 음절 2-gram 집합으로 보고 F1을 계산한다.

**왜 음절 2-gram인가** — 한국어는 조사가 붙어 어절 단위 비교가 쉽게 어긋난다.
"피한다"와 "피하고"는 어절로는 완전히 다른 토큰이지만 음절 2-gram으로 보면
"피하"를 공유한다. 형태소 분석기를 넣지 않아도 이 정도는 잡힌다. 형태소 분석기는
새 의존성이고(`docs/PACKAGES.md` 승인 대상) 학습 목적에 비해 무겁다.

**왜 F1인가** — 재현율만 쓰면 길게 쓸수록 유리해진다. 정밀도만 쓰면 짧게 쓸수록
유리해진다. 둘의 조화평균이 "필요한 것을 말하되 군더더기를 붙이지 않는" 답에
높은 점수를 준다.

**한계** — 표현이 다르고 뜻이 같은 답은 낮게 나온다. 그래서 이 점수는 절대
품질이 아니라 **기준선 대비 비교용**이다 (D-026).
"""

from __future__ import annotations

import re
from collections import Counter

# 채점 전에 지우는 문자. 띄어쓰기와 문장부호가 달라도 같은 답으로 봐야 한다.
NOISE = re.compile(r"[^0-9a-z가-힣]+")


def normalize(text: str) -> str:
    """비교에 쓰지 않는 문자를 지우고 소문자로 만든다."""
    return NOISE.sub("", text.lower())


def syllable_bigrams(text: str) -> list[str]:
    """음절 2-gram 목록. 한 글자뿐이면 그 글자를 그대로 쓴다.

    목록으로 돌려주는 이유는 빈도를 살리기 위해서다. 집합으로 만들면 같은
    2-gram을 여러 번 쓴 답이 실제보다 높은 점수를 받는다.
    """
    cleaned = normalize(text)
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return [cleaned]
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def bigram_f1(expected: str, actual: str) -> float:
    """두 문자열의 음절 2-gram F1. 범위는 [0, 1]이다.

    한쪽이라도 비어 있으면 0.0이다. 기대 답변이 없는 케이스에 1.0을 주면 채점되지
    않은 케이스가 만점으로 집계된다.
    """
    expected_grams = Counter(syllable_bigrams(expected))
    actual_grams = Counter(syllable_bigrams(actual))

    if not expected_grams or not actual_grams:
        return 0.0

    # 빈도까지 고려한 교집합. 같은 2-gram을 반복해도 기대 답변에 있는 만큼만 센다.
    overlap = sum((expected_grams & actual_grams).values())
    if overlap == 0:
        return 0.0

    precision = overlap / sum(actual_grams.values())
    recall = overlap / sum(expected_grams.values())
    return 2 * precision * recall / (precision + recall)
