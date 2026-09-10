"""도메인 값 객체.

전부 @dataclass(frozen=True)다. 불변이면 값이 어디서 바뀌었는지 추적할 필요가 없다.

검증은 __post_init__에서 한다. frozen=True라 여기서 값을 대입할 수는 없고,
불변식을 어긴 값이면 생성 자체를 실패시킨다. 이렇게 두는 이유는
"근거 없는 Answer" 같은 상태가 애초에 존재할 수 없게 만들기 위해서다 (MEMORY.md D-007).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.domain.policy import DISCLAIMER


class SkinType(Enum):
    """데이터셋의 피부 문제 유형 4종.

    문자열을 그대로 쓰면 오타가 런타임까지 간다.
    한국어 원본 표기와 영어 이름의 매핑은 여기 한 곳에만 둔다 (docs/guidelines/00-common.md).
    """

    SENSITIVE = "민감성"
    INFLAMMATORY = "염증성"
    PIGMENT = "색소문제"
    TEXTURE = "조직변화"


class RefusalReason(Enum):
    """답을 거부한 사유.

    자유 문자열 대신 Enum으로 두는 이유는, 두 게이트(SAFETY.md S-2, S-4)를
    구분해 테스트해야 하는데 문자열이면 오타가 조용히 통과하기 때문이다.

    사유가 셋인 이유는 실패 지점이 셋이기 때문이다.

    | 값 | 어디서 막혔나 | 무엇을 고쳐야 하나 |
    |---|---|---|
    | NO_EVIDENCE | 검색. 생성 호출 전이다 | 색인 재료, 임베딩, 임계값 |
    | NO_ANSWER_FROM_MODEL | 생성. 근거는 통과했다 | 프롬프트, 생성 모델 |
    | SAFETY_VIOLATION | 생성 후 검사 | 안전 규칙, 프롬프트 |

    앞의 둘을 한 값으로 묶었다가 M4-2에서 오독했다. 검색 유사도가 0.84였는데
    보고서에는 "근거 없음" 19건으로 찍혔다. 검색을 손볼 뻔했지만 실제로 고칠
    곳은 생성 쪽이었다 (MEMORY.md O-11).
    """

    NO_EVIDENCE = "근거 없음"
    NO_ANSWER_FROM_MODEL = "모델이 답을 못 만듦"
    SAFETY_VIOLATION = "안전 규칙 위반"


def _blank_to_none(instance: object, field: str) -> None:
    """빈 문자열을 None으로 바꾼다.

    실데이터에는 값이 없을 때 빈 문자열이 들어 있다. 그대로 두면 "값이 있다"로
    취급되어 메타데이터 필터링(M3-2)이 어긋난다.

    frozen=True라 일반 대입이 막힌다. 생성 시점의 정규화이므로 여기서만
    object.__setattr__을 쓴다.
    """
    value = getattr(instance, field)
    if isinstance(value, str) and not value.strip():
        object.__setattr__(instance, field, None)


@dataclass(frozen=True)
class Document:
    """지식데이터 1건. 원문과 메타데이터를 담는다."""

    id: str
    text: str
    skin_type: SkinType | None = None
    area: str | None = None
    # 세부 피부 유형(아토피 피부, 홍조 피부 등 13종). Enum이 아니라 문자열이다.
    # 실데이터에 없는 표기가 나와도 적재가 통째로 실패하지 않아야 한다 (D-028).
    skin_detail: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Document.id가 비어 있다")
        if not self.text.strip():
            raise ValueError(f"Document.text가 비어 있다 (id={self.id})")
        _blank_to_none(self, "skin_detail")


@dataclass(frozen=True)
class Chunk:
    """분할된 문단. 어느 문서에서 나왔는지를 안다.

    메타데이터를 문서에서 물려받아 다시 들고 있는 이유는, 검색 결과가 청크 단위로
    돌아오기 때문이다. 청크만 보고 피부 유형을 알 수 있어야 M3-2의 메타데이터
    필터링을 붙일 수 있다.
    """

    id: str
    doc_id: str
    text: str
    skin_type: SkinType | None = None
    area: str | None = None
    skin_detail: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Chunk.id가 비어 있다")
        if not self.doc_id.strip():
            raise ValueError(f"Chunk.doc_id가 비어 있다 (id={self.id})")
        if not self.text.strip():
            raise ValueError(f"Chunk.text가 비어 있다 (id={self.id})")
        _blank_to_none(self, "skin_detail")


@dataclass(frozen=True)
class Query:
    """사용자 질문.

    이 객체는 처리되는 동안만 존재한다. 저장하거나 로그에 남기지 않는다.
    질문 자체가 개인 건강 정보일 수 있다 (SAFETY.md S-7).
    """

    text: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Query.text가 비어 있다")


@dataclass(frozen=True)
class Evidence:
    """검색된 근거. 청크와 유사도 점수를 함께 든다."""

    chunk: Chunk
    score: float

    def __post_init__(self) -> None:
        # 코사인 유사도는 [-1, 1]을 벗어날 수 없다. 벗어났다면 색인 쪽 계산 버그다.
        # 여기서 막지 않으면 잘못된 점수가 RetrievalPolicy의 임계값 판정까지 흘러간다.
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"유사도가 [-1, 1] 밖이다: {self.score}")


@dataclass(frozen=True)
class Answer:
    """생성된 답.

    두 가지 불변식을 생성 시점에 강제한다.

    1. 근거가 없으면 만들 수 없다. D-007은 application의 if 문으로도 막지만,
       값 객체에서도 막아 두면 새 호출 경로가 생겨도 뚫리지 않는다.
    2. 면책 문구는 필드가 아니라 text 프로퍼티가 붙인다. 필드로 두면 빈 문자열을
       넣을 수 있고, 그러면 S-5의 "조건부로 붙이지 않는다"가 깨진다.
    """

    body: str
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        if not self.body.strip():
            raise ValueError("Answer.body가 비어 있다")
        # frozen=True인데 필드가 list면 evidence.append()로 내용이 바뀐다.
        # 불변이 이름뿐이 되므로 tuple만 받는다.
        if not isinstance(self.evidence, tuple):
            raise TypeError(
                f"Answer.evidence는 tuple이어야 한다 (받은 타입: {type(self.evidence).__name__}). "
                "list를 넘겼다면 tuple(...)로 감싸라"
            )
        if not self.evidence:
            raise ValueError("근거 없는 Answer는 만들 수 없다 (MEMORY.md D-007)")

    @property
    def text(self) -> str:
        """밖으로 내보내는 최종 문자열. 면책 문구가 항상 붙는다 (SAFETY.md S-5)."""
        return f"{self.body.rstrip()}\n\n{DISCLAIMER}"


@dataclass(frozen=True)
class Refusal:
    """답할 수 없음.

    예외가 아니라 값으로 표현한다. 예외는 안 잡으면 흘러가지만 값은 반드시
    분기해야 한다 (docs/ARCHITECTURE.md 3절).
    """

    reason: RefusalReason
    detail: str = ""

    def __post_init__(self) -> None:
        # 문자열을 그대로 넘기는 실수를 여기서 잡는다. 통과시키면 사유별 분기가
        # 조용히 어긋난다.
        if not isinstance(self.reason, RefusalReason):
            raise TypeError(
                f"Refusal.reason은 RefusalReason이어야 한다 "
                f"(받은 타입: {type(self.reason).__name__})"
            )


@dataclass(frozen=True)
class EvalCase:
    """평가용 질문 1건. QA쌍 하나를 이 형태로 바꿔 쓴다 (MEMORY.md D-008).

    gold_doc_ids는 이 질문의 근거가 있어야 할 문서 ID다. Recall@k는 검색 결과에
    이 중 하나라도 들어 있는지로 판정한다.

    expected는 기대 답변 문장이다. 채점 방식이 아직 정해지지 않았으므로(M2-7)
    지금은 보관만 한다.
    """

    id: str
    question: str
    expected: str = ""
    gold_doc_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("EvalCase.id가 비어 있다")
        if not self.question.strip():
            raise ValueError(f"EvalCase의 질문이 비어 있다 (id={self.id})")
        # frozen=True인데 필드가 list면 내용이 바뀐다. Answer.evidence와 같은 이유다.
        if not isinstance(self.gold_doc_ids, tuple):
            raise TypeError(
                f"EvalCase.gold_doc_ids는 tuple이어야 한다 "
                f"(받은 타입: {type(self.gold_doc_ids).__name__})"
            )

    @property
    def is_gradable(self) -> bool:
        """Recall을 계산할 수 있는가.

        QA쌍이 근거 문단 ID를 갖지 않을 수 있다(M2-1에서 확인). 근거가 없으면
        맞았는지 틀렸는지 판정할 수 없다. 0점으로 세면 점수가 실제보다 낮게
        나오므로 채점 대상에서 뺀다.
        """
        return bool(self.gold_doc_ids)


# 호출부가 두 경우를 반드시 분기하게 만든다. 근거 없는 생성을 타입 수준에서 막는 장치다.
Result = Answer | Refusal
