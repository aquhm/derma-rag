"""평가 유스케이스. 검색 채점과 답변 채점을 분리한다.

둘을 나누는 이유는 `MEMORY.md` D-008에 있다. Recall@k가 낮으면 검색을 고쳐야
하고, Recall이 높은데 답이 틀리면 생성이나 프롬프트를 고쳐야 한다. 하나의
점수로 합치면 어느 쪽인지 알 수 없다.

**Recall@k 정의** — 상위 k개 검색 결과의 문서 ID 중 정답 문서 ID가 하나라도
있으면 hit로 센다 (MEMORY.md D-025). 정답 문서가 여럿일 때 몇 개를 찾았는지는
세지 않는다.

**보고서에 질문 원문을 담지 않는다.** 케이스 ID와 숫자만 남긴다. 질문은 개인
건강 정보일 수 있다 (SAFETY.md S-7).

**답변 정확도** — 기대 답변과 생성 답변의 음절 2-gram F1이다 (MEMORY.md D-026,
`src/domain/scoring.py`). 거부한 케이스는 정확도 계산에서 뺀다. 거부는 틀린 답이
아니라 답하지 않은 것이고, 거부율로 이미 세고 있다.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from src.application.ports import Embedder, VectorIndex
from src.domain.models import Answer, EvalCase, Query, Refusal
from src.domain.scoring import bigram_f1

# (처리한 건수, 전체 건수). 출력은 interface의 일이므로 호출자에게 넘긴다.
ProgressCallback = Callable[[int, int], None]


class QuestionAnswerer(Protocol):
    """AnswerQuestion이 만족하는 계약.

    유스케이스를 유스케이스에 주입한다. 둘 다 application이라 의존 방향이
    뒤집히지 않는다. Protocol로 두면 평가 테스트에서 가짜를 쓸 수 있다.
    """

    def execute(self, query: Query) -> Answer | Refusal: ...


@dataclass(frozen=True)
class CaseResult:
    """케이스 1건의 검색 결과 요약. 질문 원문은 담지 않는다."""

    case_id: str
    hit: bool | None
    best_score: float
    retrieved_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    results: tuple[CaseResult, ...]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def graded(self) -> int:
        """정답 문서 ID가 있어 채점할 수 있었던 건수."""
        return sum(1 for r in self.results if r.hit is not None)

    @property
    def hits(self) -> int:
        return sum(1 for r in self.results if r.hit)

    @property
    def recall(self) -> float:
        """채점 가능한 건수 기준 Recall@k. 채점할 것이 없으면 0.0."""
        return self.hits / self.graded if self.graded else 0.0

    @property
    def mean_best_score(self) -> float:
        """케이스별 최고 유사도의 평균.

        임계값(min_score)을 정하려면 점수가 실제로 어느 대역에 있는지 알아야
        한다. D-021이 남긴 숙제다.
        """
        if not self.results:
            return 0.0
        return sum(r.best_score for r in self.results) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        """eval/ 아래 파일에 넣을 형태 (eval/README.md 형식)."""
        return {
            f"recall_at_{self.k}": round(self.recall, 4),
            "sample_count": self.total,
            "graded_count": self.graded,
            "hit_count": self.hits,
            "mean_best_score": round(self.mean_best_score, 4),
        }


@dataclass(frozen=True)
class CaseOutcome:
    """케이스 1건의 결과. 질문도 답변도 담지 않는다 (SAFETY.md S-7).

    실험끼리 공통 케이스만 골라 비교하려면 케이스별 점수가 있어야 한다. 요약
    숫자만 남기면 채점 대상이 달라졌을 때 두 실험을 견줄 수 없다 (미해결 O-5).
    """

    case_id: str
    # answered | refused | failed
    outcome: str
    accuracy: float | None = None
    similarity: float | None = None
    # 거부 사유 또는 예외 종류. 원문은 넣지 않는다.
    detail: str = ""


@dataclass(frozen=True)
class AnswerReport:
    total: int
    answered: int
    by_reason: dict[str, int]
    # 기대 답변이 있고 실제로 답한 케이스의 점수만 담는다.
    accuracy_scores: tuple[float, ...] = ()
    similarity_scores: tuple[float, ...] | None = None
    # 생성이 실패해 판정하지 못한 케이스. 예외 종류만 담는다 (ERRORS.md E-003).
    failures: tuple[str, ...] = ()
    # 케이스별 결과. eval/raw/에 저장해 실험 간 공통 집합 비교에 쓴다 (O-5).
    cases: tuple[CaseOutcome, ...] = ()

    @property
    def failed(self) -> int:
        return len(self.failures)

    @property
    def refused(self) -> int:
        return self.total - self.answered - self.failed

    @property
    def completed(self) -> int:
        """끝까지 돈 케이스 수. 거부율의 분모다."""
        return self.answered + self.refused

    @property
    def refusal_rate(self) -> float:
        """거부율. 실패한 케이스는 분모에서 뺀다.

        실패는 시스템 장애고 거부는 정상 동작이다. 섞으면 Ollama가 흔들릴 때
        거부율이 좋아 보이거나 나빠 보인다.
        """
        return self.refused / self.completed if self.completed else 0.0

    @property
    def scored(self) -> int:
        """정확도를 잰 케이스 수. 기대 답변이 없거나 거부된 건은 빠진다."""
        return len(self.accuracy_scores)

    @property
    def accuracy(self) -> float:
        """음절 2-gram F1의 평균. 잰 것이 없으면 0.0."""
        return _mean(self.accuracy_scores)

    @property
    def similarity(self) -> float:
        """임베딩 코사인의 평균. 보조 지표다 (D-026)."""
        return _mean(self.similarity_scores or ())

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "answer_accuracy": round(self.accuracy, 4),
            "scored_count": self.scored,
            "refusal_rate": round(self.refusal_rate, 4),
            "answered_count": self.answered,
            "refused_count": self.refused,
            "refusal_reasons": dict(self.by_reason),
            "failed_count": self.failed,
            "sample_count": self.total,
        }
        if self.similarity_scores is not None:
            payload["answer_similarity"] = round(self.similarity, 4)
        return payload


class EvaluateRetrieval:
    """검색 단계만 채점한다. 생성 모델을 부르지 않는다."""

    def __init__(self, embedder: Embedder, index: VectorIndex, k: int = 5) -> None:
        if k < 1:
            raise ValueError(f"k는 1 이상이어야 한다: {k}")
        self._embedder = embedder
        self._index = index
        self._k = k

    def execute(
        self, cases: list[EvalCase], on_progress: ProgressCallback | None = None
    ) -> RetrievalReport:
        if not cases:
            raise ValueError("평가 케이스가 없다")

        # 질문을 한 번에 임베딩한다. 어댑터가 내부에서 64건씩 나눠 보낸다(D-016).
        vectors = self._embedder.embed([c.question for c in cases])
        if len(vectors) != len(cases):
            raise ValueError(
                f"질문 {len(cases)}건에 벡터 {len(vectors)}건이 돌아왔다"
            )

        results = []
        for position, (case, vector) in enumerate(zip(cases, vectors), start=1):
            results.append(self._grade(case, vector))
            if on_progress is not None:
                on_progress(position, len(cases))

        return RetrievalReport(k=self._k, results=tuple(results))

    def _grade(self, case: EvalCase, vector: list[float]) -> CaseResult:
        found = self._index.search(vector, self._k)
        doc_ids = tuple(item.chunk.doc_id for item in found)
        best = max((item.score for item in found), default=0.0)

        return CaseResult(
            case_id=case.id,
            # 정답 문서 ID가 없으면 판정하지 않는다. 0점으로 세면 점수가 실제보다
            # 낮게 나온다 (EvalCase.is_gradable).
            hit=bool(set(case.gold_doc_ids) & set(doc_ids)) if case.is_gradable else None,
            best_score=best,
            retrieved_doc_ids=doc_ids,
        )


class EvaluateAnswers:
    """답변 경로 전체를 돌려 정확도, 거부율, 거부 사유 분포를 낸다.

    embedder를 주면 보조 지표로 임베딩 코사인도 함께 낸다. 주 지표는 음절
    2-gram F1이다 (D-026). 코사인은 표현이 달라도 뜻이 같은 답을 F1이 얼마나
    낮게 보는지 견주는 용도다.
    """

    def __init__(
        self, answerer: QuestionAnswerer, embedder: Embedder | None = None
    ) -> None:
        self._answerer = answerer
        self._embedder = embedder

    def execute(
        self, cases: list[EvalCase], on_progress: ProgressCallback | None = None
    ) -> AnswerReport:
        if not cases:
            raise ValueError("평가 케이스가 없다")

        answered = 0
        reasons: Counter[str] = Counter()
        scored: list[tuple[str, str]] = []
        failures: list[str] = []
        rows: list[CaseOutcome] = []
        # 유사도는 채점 대상 전체를 한 번에 임베딩한 뒤에야 나온다. 그 전까지
        # 어느 행에 넣을지 기억해 둔다.
        scored_rows: list[int] = []

        for position, case in enumerate(cases, start=1):
            try:
                result = self._answerer.execute(Query(case.question))
            except Exception as exc:  # noqa: BLE001 - 아래 주석 참조
                # 200건 평가는 20분 걸린다. 한 건이 타임아웃 났다고 전부 잃으면
                # 안 된다 (ERRORS.md E-003). 예외 종류와 케이스 ID만 남긴다.
                # 질문은 남기지 않는다 (SAFETY.md S-7).
                #
                # 예외 타입을 좁히지 않는 이유는 여기가 application이기 때문이다.
                # 어떤 어댑터가 어떤 예외를 던지는지 알면 의존 방향이 뒤집힌다.
                failures.append(f"{case.id}: {type(exc).__name__}")
                rows.append(
                    CaseOutcome(case.id, "failed", detail=type(exc).__name__)
                )
                if on_progress is not None:
                    on_progress(position, len(cases))
                continue

            if isinstance(result, Refusal):
                reasons[result.reason.value] += 1
                rows.append(
                    CaseOutcome(case.id, "refused", detail=result.reason.value)
                )
            else:
                answered += 1
                accuracy = None
                if case.expected.strip():
                    accuracy = bigram_f1(case.expected, result.body)
                    scored.append((case.expected, result.body))
                    scored_rows.append(len(rows))
                rows.append(CaseOutcome(case.id, "answered", accuracy=accuracy))
            if on_progress is not None:
                on_progress(position, len(cases))

        similarities = self._similarities(scored)
        if similarities:
            for position, similarity in zip(scored_rows, similarities):
                rows[position] = replace(rows[position], similarity=similarity)

        return AnswerReport(
            total=len(cases),
            answered=answered,
            by_reason=dict(reasons),
            accuracy_scores=tuple(bigram_f1(e, a) for e, a in scored),
            similarity_scores=similarities,
            failures=tuple(failures),
            cases=tuple(rows),
        )

    def _similarities(self, scored: list[tuple[str, str]]) -> tuple[float, ...] | None:
        """기대 답변과 생성 답변의 코사인. embedder가 없으면 None."""
        if self._embedder is None:
            return None
        if not scored:
            return ()

        # 기대 답변과 생성 답변을 한 번에 임베딩한다. 어댑터가 내부에서 나눠 보낸다.
        texts = [t for pair in scored for t in pair]
        vectors = self._embedder.embed(texts)
        return tuple(
            _cosine(vectors[i], vectors[i + 1]) for i in range(0, len(vectors), 2)
        )


def sample_cases(cases: list[EvalCase], size: int, seed: int) -> list[EvalCase]:
    """평가셋에서 표본을 뽑는다. size가 0 이하면 전량을 쓴다.

    seed를 고정하는 이유는 실험 간 비교 때문이다. 실험마다 다른 표본을 쓰면
    점수 차이가 변경 때문인지 표본 때문인지 구분되지 않는다 (D-026).

    원래 순서를 유지한다. 결과 파일을 눈으로 훑을 때 케이스 순서가 뒤섞이면
    같은 표본인지 확인하기 어렵다.
    """
    if size <= 0 or size >= len(cases):
        return list(cases)

    chosen = set(random.Random(seed).sample(range(len(cases)), size))
    return [case for position, case in enumerate(cases) if position in chosen]


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cosine(left: list[float], right: list[float]) -> float:
    """코사인 유사도. numpy를 쓰지 않는 이유는 여기가 application이기 때문이다."""
    norm = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(left, right)) / norm
