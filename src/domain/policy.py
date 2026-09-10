"""도메인 정책. 판정 규칙과 고정 문구를 담는다.

정책을 값 객체로 빼는 이유는 임계값 조정이 코드 수정이 아니라 값 변경이 되게
하기 위해서다. M3-3에서 k와 min_score를 바꿔 가며 측정한다
(docs/guidelines/01-domain.md).

주의 — Evidence는 models.py에 있고 models.py가 이 모듈의 DISCLAIMER를 import
한다. 런타임에 여기서 models를 import 하면 순환 import가 된다. 타입 힌트에만
쓰이므로 TYPE_CHECKING 블록 안에서만 가져온다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.models import Evidence

# SAFETY.md S-5. Answer가 반환되는 모든 경로에 붙는다. 조건부로 붙이지 않는다.
# 출처 표기는 안전 요건이자 데이터셋 이용 조건이다 (docs/DATASETS.md).
DISCLAIMER = (
    "이 답변은 AI Hub 「문제성 피부 메이크업 추천 데이터」를 검색한 결과이며\n"
    "의학적 진단이나 처방이 아닙니다.\n"
    "증상이 지속되거나 악화되면 피부과 전문의와 상담하십시오.\n"
    "데이터 출처: aihub.or.kr"
)


@dataclass(frozen=True)
class RetrievalPolicy:
    """검색 결과가 답하기에 충분한지 판정한다 (SAFETY.md S-2).

    k=3은 M3-3 측정으로 정했다. k=5에서 3으로 줄이면 F1이 +0.0053 올랐고
    (공통 197건, 표준오차 0.0031), k=1까지 줄여도 더 오르지 않았다. 4~5번째
    근거는 노이즈다. k=1과 k=3이 같은 점수라면 사용자에게 보여 줄 근거가 많은
    쪽을 택한다.

    min_score=0.35는 사실상 게이트로 동작하지 않는다. 실데이터 최고 유사도가
    0.638~0.780에 뭉쳐 있어 어떤 임계값도 좋은 근거와 나쁜 근거를 가르지 못한다
    (MEMORY.md D-030).
    """

    k: int = 3
    min_score: float = 0.35

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k는 1 이상이어야 한다: {self.k}")
        # 코사인 유사도의 범위 밖 임계값은 항상 통과하거나 항상 거부한다.
        # 둘 다 게이트가 없는 것과 같다.
        if not -1.0 <= self.min_score <= 1.0:
            raise ValueError(f"min_score는 [-1, 1] 안이어야 한다: {self.min_score}")

    def is_sufficient(self, evidence: list[Evidence]) -> bool:
        """임계값을 넘는 근거가 하나라도 있는가.

        정렬을 전제하지 않는다. 색인이 정렬해서 주더라도 그 약속에 기대면 다른
        색인 구현으로 바꿀 때 조용히 깨진다.
        """
        return bool(self.select(evidence))

    def select(self, evidence: list[Evidence]) -> tuple[Evidence, ...]:
        """프롬프트와 Answer에 실을 근거만 고른다.

        임계값 미만을 빼는 이유는 두 가지다. 관련 없는 문단이 생성 모델을 흔든다.
        그리고 Answer.evidence는 사용자에게 근거로 제시되므로, 점수가 낮은 문단이
        섞이면 답변과 무관한 문단을 근거라고 보여 주게 된다.

        입력 순서를 유지한다. 색인이 점수 내림차순으로 주므로 결과도 그 순서다.
        """
        return tuple(e for e in evidence if e.score >= self.min_score)


@dataclass(frozen=True)
class SafetyPolicy:
    """생성 결과에 진단·치료 단정이 섞였는지 검사한다 (SAFETY.md S-1, S-4).

    프롬프트로도 같은 지시를 하지만 프롬프트는 어길 수 있다. 이 검사가 두 번째
    게이트다.

    규칙은 (이름, 정규식) 쌍이다. 이름은 Refusal.detail로 나가므로 생성 원문을
    담지 않는다 (docs/guidelines/00-common.md 로그 항목).
    """

    rules: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name, pattern in self.rules:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"규칙 '{name}'의 정규식이 잘못됐다: {exc}") from exc

    @classmethod
    def default(cls) -> SafetyPolicy:
        """기본 금지 표현 목록.

        넓게 잡으면 정상 답변까지 거부한다. 예를 들어 "~입니다" 전체를 막으면
        모든 답이 막힌다. 그래서 질환명과 붙은 단정, 완치·치료 단정, 처방·복약
        지시, 의료 이용 판단만 잡는다. 데이터셋 문장에 흔한 "~라고 기술돼 있습니다",
        "~하는 편이 낫다고 설명된다"는 통과해야 한다.

        "의료 이용 판단"은 D-034에서 추가했다. 실사용 확인에서 "이 증상 병원 안
        가도 되나요?"에 시스템이 답하는 것을 봤다. 진단을 안 해도 진료를 미루게
        만들 수 있다.
        """
        diseases = (
            "여드름|지루성\\s?피부염|아토피(?:\\s?피부염)?|건선|습진|주사비|모낭염|"
            "접촉성\\s?피부염|백선|무좀|피부암|종양|피부염"
        )
        return cls(
            rules=(
                (
                    "진단 단정",
                    rf"(?:{diseases})\s*(?:[이가]\s*)?"
                    r"(?:입니다|이다|이에요|예요|맞습니다|확실합니다|확실해요|"
                    r"분명합니다|으로\s*보입니다|로\s*보입니다|같습니다|"
                    r"(?:으)?로\s*진단)",
                ),
                (
                    "치료 단정",
                    r"완치|치료됩니다|치료된다|치료돼요|나았습니다|낫습니다|나아집니다|"
                    r"없어집니다|사라집니다|해결됩니다",
                ),
                (
                    "처방·복약 지시",
                    r"처방(?:해|을|합니다|받으세요|해\s*드립니다)|"
                    r"복용(?:하세요|하십시오|하시면|을\s*권)|약을\s*(?:드시|먹)",
                ),
                (
                    "효능 단정",
                    r"(?:바르면|사용하면|쓰면|먹으면)\s*(?:낫|좋아|사라|없어|치료)",
                ),
                (
                    "의료 이용 판단",
                    # "병원 안 가도 된다"는 진단만큼 위험하다. 사용자가 그 말을
                    # 믿고 진료를 미룰 수 있다 (D-034).
                    #
                    # 병원을 권하는 문장은 통과해야 한다. 면책 문구에
                    # "피부과 전문의와 상담하십시오"가 들어 있고, 그것까지 막으면
                    # 모든 Answer가 거부된다.
                    r"(?:병원|피부과|진료|치료)[^.\n]{0,10}?"
                    r"(?:안\s*가도|가지\s*않아도|필요하지\s*않|안\s*받아도)"
                    r"|심각(?:한\s*상태는\s*아닙니다|하지\s*않습니다|하지\s*않아요)"
                    r"|걱정(?:하지\s*않으셔도|할\s*필요\s*없|안\s*하셔도)"
                    r"|\d+\s*(?:일|주|개월|달)\s*(?:정도\s*)?(?:면|이면|안에)"
                    r"\s*(?:좋아|나아|낫|괜찮|호전)",
                ),
            )
        )

    def violation(self, text: str) -> str | None:
        """위반한 규칙의 이름. 위반이 없으면 None.

        문자열이 아니라 이름만 돌려주는 이유는 호출부가 사유별로 분기하거나
        기록할 때 생성 원문을 들고 다니지 않게 하기 위해서다.
        """
        for name, pattern in self.rules:
            if re.search(pattern, text):
                return name
        return None
