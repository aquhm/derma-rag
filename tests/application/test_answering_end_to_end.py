"""M1-6과 M1-7의 완료 조건을 실제 스택으로 확인한다.

가짜 없이 로더 → 청커 → 임베딩 → 색인 → 검색 → 생성을 전부 통과시킨다. Ollama가
떠 있어야 하므로 integration으로 표시한다.

여기서 확인하는 것은 두 가지다.

- M1-6: 질문 1건에 근거를 포함한 Answer가 돌아온다.
- M1-7: 무관한 질문에는 Refusal이 돌아온다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.answering import AnswerQuestion
from src.domain.models import Answer, Query, Refusal, RefusalReason
from src.domain.policy import DISCLAIMER, RetrievalPolicy, SafetyPolicy
from src.infrastructure.chunking import ParagraphChunker
from src.infrastructure.embedding import OllamaEmbedder
from src.infrastructure.generation import OllamaGenerator
from src.infrastructure.index import NumpyVectorIndex
from src.infrastructure.loaders import JsonDocumentLoader

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "documents.json"


@pytest.fixture(scope="module")
def usecase() -> AnswerQuestion:
    """샘플 문서 10건으로 실제 색인을 만든다. 임베딩 호출을 아끼려고 모듈 단위다."""
    chunks = [
        c
        for doc in JsonDocumentLoader(SAMPLES).load()
        for c in ParagraphChunker().split(doc)
    ]
    embedder = OllamaEmbedder()
    index = NumpyVectorIndex()
    index.add(chunks, embedder.embed([c.text for c in chunks]))

    return AnswerQuestion(
        embedder=embedder,
        index=index,
        generator=OllamaGenerator(),
        retrieval_policy=RetrievalPolicy(),
        safety_policy=SafetyPolicy.default(),
    )


@pytest.mark.integration
def test_relevant_question_gets_an_answer_with_evidence(
    usecase: AnswerQuestion,
) -> None:
    """M1-6 완료 조건."""
    result = usecase.execute(Query("민감성 피부는 메이크업 전에 무엇을 주의해야 하나요?"))

    assert isinstance(result, Answer), getattr(result, "detail", "")
    assert result.evidence
    assert DISCLAIMER in result.text


@pytest.mark.integration
@pytest.mark.parametrize(
    "question",
    [
        "자동차 엔진 오일은 몇 km마다 교환해야 하나요?",
        "파이썬에서 리스트를 정렬하는 방법을 알려주세요.",
    ],
)
def test_unrelated_question_is_refused(usecase: AnswerQuestion, question: str) -> None:
    """M1-7 완료 조건.

    현재 이 거부는 검색 게이트가 아니라 생성 결과 판정에서 난다.
    nomic-embed-text의 코사인 유사도가 무관한 질문에서도 0.72~0.84로 높아
    min_score=0.35 게이트가 걸리지 않는다 (2026-09-08 실측, MEMORY.md D-021).
    임계값은 실데이터로 Recall을 측정한 뒤 M3-3에서 조정한다.
    """
    result = usecase.execute(Query(question))

    assert isinstance(result, Refusal), getattr(result, "body", "")
    assert result.reason is RefusalReason.NO_EVIDENCE
