"""AnswerQuestion 유스케이스. 5단계(검색)와 6단계(생성)를 잇는다.

두 게이트를 순서대로 통과해야 Answer가 나온다 (SAFETY.md S-2, S-4).

    검색 → RetrievalPolicy.is_sufficient()
            ├ False → Refusal(NO_EVIDENCE). Generator를 부르지 않는다
            └ True  → 프롬프트 조립 → 생성 → SafetyPolicy.violation()
                                              ├ 있음 → Refusal(SAFETY_VIOLATION)
                                              └ 없음 → Answer

게이트를 프롬프트 문구로 대체하지 않는다. 프롬프트는 어길 수 있고 if 문은 못
어긴다 (docs/guidelines/02-application.md).

프롬프트 문자열은 여기서 만든다. 어떤 모델인지는 모른 채로 만들고, 모델별 요청
포맷 변환은 어댑터가 한다.
"""

from __future__ import annotations

import re

from src.application.ports import Embedder, Generator, VectorIndex
from src.domain.models import (
    Answer,
    Evidence,
    Query,
    Refusal,
    RefusalReason,
    Result,
)
from src.domain.policy import RetrievalPolicy, SafetyPolicy

# 모델이 "자료에 답이 없다"고 알릴 때 쓰게 할 고정 토큰.
#
# 처음에는 한국어 문장("제공된 자료에 해당 내용이 없습니다")을 시켰다. 실제로는
# qwen2.5:3b가 문장을 매번 다르게 바꿔 써서 문자열 비교가 실패했고, 근거 없음이
# Answer로 나갔다 (2026-09-08 실측, ERRORS.md E-002). 짧은 영문 토큰은 모델이
# 그대로 뱉을 확률이 높고 비교가 정확하다.
NO_ANSWER_TOKEN = "NO_ANSWER"

# 토큰 지시를 어기고 한국어로 풀어 쓴 경우를 잡는 보조 검사.
#
# 주어를 "자료·문단·문서"로 제한한다. "없습니다"만 보면 "알코올이 없습니다" 같은
# 정상 답변까지 거부한다.
NO_ANSWER_PATTERN = re.compile(
    r"(?:제공된|주어진|위의|해당)?\s*(?:자료|문단|문서)(?:에는|에서|에|의)"
    r"[^.\n]{0,40}?"
    r"(?:없습니다|없다|없음|찾을\s*수\s*없습니다|찾지\s*못했습니다|확인할\s*수\s*없습니다)"
)

# SAFETY.md S-3. 사전 지식으로 보충하는 것을 막는 지시다. 이것만으로 충분하지
# 않으므로 생성 뒤에 SafetyPolicy 검사가 붙는다.
#
# 길이 제약은 M3-4에서 붙였다. 기준선에서 생성 답변이 기대 답변의 1.51배 길었고
# (644자 대 456자), 음절 2-gram F1은 장황한 답에 정밀도 벌점을 준다. 예비 측정
# 30건에서 F1 0.231 → 0.243이었다.
INSTRUCTION = f"""당신은 제공된 문단만 근거로 답하는 검색 보조자입니다.

규칙:
- 아래 제공된 문단에 없는 내용은 답하지 마십시오.
- 문단에 답이 없으면 다른 말을 덧붙이지 말고 {NO_ANSWER_TOKEN} 한 줄만 출력하십시오.
- 추측하거나 일반 상식으로 보충하지 마십시오.
- 진단하지 마십시오. "~입니다" 같은 단정 대신 "자료에는 ~라고 기술돼 있습니다"로 쓰십시오.
- 한국어로 답하십시오.
- 400자 이내, 3~5문장으로 답하십시오."""


class AnswerQuestion:
    """질문 1건을 받아 Answer 또는 Refusal을 돌려준다.

    의존은 전부 생성자로 받는다. 조립은 composition.py에서만 한다
    (docs/ARCHITECTURE.md 5절).
    """

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        generator: Generator,
        retrieval_policy: RetrievalPolicy,
        safety_policy: SafetyPolicy,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._generator = generator
        self._retrieval = retrieval_policy
        self._safety = safety_policy

    def execute(self, query: Query) -> Result:
        if not isinstance(query, Query):
            # 문자열을 그대로 받으면 Query의 빈 문자열 검사를 건너뛴다.
            raise TypeError(
                f"Query가 필요하다 (받은 타입: {type(query).__name__})"
            )

        evidence = self._search(query)

        # 첫 번째 게이트. 여기서 끝나면 생성 호출이 없다 (MEMORY.md D-007).
        selected = self._retrieval.select(evidence)
        if not selected:
            return Refusal(
                reason=RefusalReason.NO_EVIDENCE,
                detail=self._why_not_enough(evidence),
            )

        body = self._generator.generate(build_prompt(query, selected)).strip()

        if not body or _says_no_answer(body):
            # 모델이 "자료에 답이 없다"고 알린 경우다. 근거 문단은 통과했지만 답은
            # 없다. 이것을 Answer로 내보내면 "내용이 없습니다"가 근거 5건과 함께
            # 답변으로 나간다.
            #
            # 검색 실패(위 분기)와 사유를 나눈다. 묶어 두면 평가 보고서를 보고
            # 검색을 고쳐야 할지 생성을 고쳐야 할지 알 수 없다 (O-11).
            return Refusal(
                reason=RefusalReason.NO_ANSWER_FROM_MODEL,
                detail="생성 결과에 답이 없다",
            )

        # 두 번째 게이트. 프롬프트 지시를 어긴 출력을 여기서 막는다 (SAFETY.md S-4).
        violation = self._safety.violation(body)
        if violation is not None:
            return Refusal(reason=RefusalReason.SAFETY_VIOLATION, detail=violation)

        return Answer(body=body, evidence=selected)

    def _why_not_enough(self, evidence: list[Evidence]) -> str:
        """거부 사유. 최고 점수와 임계값을 함께 담는다.

        CLI가 이 문자열을 그대로 보여 준다. 학습 목적이므로 왜 거부됐는지가
        숫자로 보여야 한다 (docs/guidelines/04-interface.md). 청크 내용은 담지
        않는다.
        """
        if not evidence:
            return "검색 결과가 없다"
        best = max(item.score for item in evidence)
        return f"최고 유사도 {best:.2f} < 임계값 {self._retrieval.min_score}"

    def _search(self, query: Query) -> list[Evidence]:
        vectors = self._embedder.embed([query.text])
        if not vectors:
            # 어댑터가 계약을 어긴 경우다. 빈 목록으로 검색하면 원인이 검색 실패로
            # 보이게 되므로 여기서 끊는다.
            raise ValueError("질문 벡터를 얻지 못했다")
        return self._index.search(vectors[0], self._retrieval.k)


def build_prompt(query: Query, evidence: tuple[Evidence, ...]) -> str:
    """지시 + 근거 문단 + 질문으로 프롬프트를 만든다.

    넣는 것은 검색된 근거뿐이다. 색인 전체나 모델의 사전 지식에 기대는 문구를
    넣지 않는다 (SAFETY.md S-3).
    """
    passages = "\n\n".join(
        f"[문단 {number}] {_label(item)}\n{item.chunk.text}"
        for number, item in enumerate(evidence, start=1)
    )
    return (
        f"{INSTRUCTION}\n\n"
        f"제공된 문단:\n{passages}\n\n"
        f"질문: {query.text}\n"
        f"답변:"
    )


def _label(item: Evidence) -> str:
    """문단 머리에 붙는 출처 표시.

    청크 ID를 넣는 이유는 답변에서 어느 문단을 썼는지 사람이 되짚을 수 있게 하기
    위해서다. 피부 유형과 부위는 있을 때만 붙인다. 없는 값을 "미상"으로 채우면
    모델이 그것을 사실로 받아들인다.
    """
    parts = [item.chunk.id]
    if item.chunk.skin_type is not None:
        parts.append(item.chunk.skin_type.value)
    if item.chunk.area is not None:
        parts.append(item.chunk.area)
    return f"({' · '.join(parts)})"


def _says_no_answer(body: str) -> bool:
    """모델이 "자료에 답이 없다"고 알렸는가.

    토큰을 먼저 보고, 지시를 어기고 한국어로 풀어 쓴 경우를 정규식으로 받는다.
    """
    return NO_ANSWER_TOKEN in body or NO_ANSWER_PATTERN.search(body) is not None
