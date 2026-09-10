"""OllamaEmbedder 테스트.

대부분은 가짜 세션으로 돌린다. 실제 Ollama가 필요한 것만 integration으로 표시한다.

    pytest -q -m "not integration"   # Ollama 없이
    pytest -q -m integration          # Ollama 필요
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src.infrastructure.chunking import ParagraphChunker
from src.infrastructure.embedding import OllamaEmbedder
from src.infrastructure.errors import EmbeddingUnavailable
from src.infrastructure.loaders import JsonDocumentLoader

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "documents.json"


class FakeResponse:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        return self._payload


class FakeSession:
    """requests.Session 대신 쓰는 가짜. 호출 인자를 기록한다.

    unittest.mock을 쓰지 않는 이유는 시그니처가 어긋나도 조용히 통과하기 때문이다
    (docs/guidelines/05-testing.md).
    """

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        if not self._responses:
            raise AssertionError("준비된 응답보다 요청이 많다")
        return self._responses.pop(0)


def vectors(count: int, dim: int = 3) -> list[list[float]]:
    return [[float(i)] * dim for i in range(count)]


def test_empty_input_returns_empty_without_calling_ollama() -> None:
    session = FakeSession([])

    assert OllamaEmbedder(session=session).embed([]) == []
    assert session.requests == []


def test_returns_one_vector_per_text() -> None:
    session = FakeSession([FakeResponse({"embeddings": vectors(2)})])

    result = OllamaEmbedder(session=session).embed(["첫 문장", "둘째 문장"])

    assert len(result) == 2
    assert result[0] == [0.0, 0.0, 0.0]


def test_splits_into_batches() -> None:
    session = FakeSession(
        [
            FakeResponse({"embeddings": vectors(2)}),
            FakeResponse({"embeddings": vectors(2)}),
            FakeResponse({"embeddings": vectors(1)}),
        ]
    )

    result = OllamaEmbedder(batch_size=2, session=session).embed(["가", "나", "다", "라", "마"])

    assert len(result) == 5
    assert len(session.requests) == 3
    assert [len(r["json"]["input"]) for r in session.requests] == [2, 2, 1]


def test_sends_model_and_timeout() -> None:
    session = FakeSession([FakeResponse({"embeddings": vectors(1)})])

    OllamaEmbedder(model="테스트모델", timeout=12.5, session=session).embed(["문장"])

    sent = session.requests[0]
    assert sent["json"]["model"] == "테스트모델"
    assert sent["timeout"] == 12.5
    assert sent["url"].endswith("/api/embed")


def test_network_failure_becomes_domain_error() -> None:
    """requests 예외가 위로 새어 나가지 않는다."""
    session = FakeSession([FakeResponse({}, error=requests.ConnectionError("연결 거부"))])

    with pytest.raises(EmbeddingUnavailable):
        OllamaEmbedder(session=session).embed(["문장"])


def test_error_message_does_not_leak_input_text() -> None:
    """예외 메시지에 원본 텍스트를 넣지 않는다 (00-common.md 로그 항목)."""
    secret = "각질층 장벽이 손상되면"
    session = FakeSession([FakeResponse({}, error=requests.ConnectionError("연결 거부"))])

    with pytest.raises(EmbeddingUnavailable) as caught:
        OllamaEmbedder(session=session).embed([secret])

    assert secret not in str(caught.value)


def test_missing_embeddings_key_is_rejected() -> None:
    session = FakeSession([FakeResponse({"embedding": [1.0, 2.0]})])

    with pytest.raises(EmbeddingUnavailable, match="embeddings가 없다"):
        OllamaEmbedder(session=session).embed(["문장"])


def test_vector_count_mismatch_is_rejected() -> None:
    session = FakeSession([FakeResponse({"embeddings": vectors(1)})])

    with pytest.raises(EmbeddingUnavailable, match="벡터 수가 맞지 않는다"):
        OllamaEmbedder(session=session).embed(["첫 문장", "둘째 문장"])


def test_inconsistent_dimensions_are_rejected() -> None:
    session = FakeSession([FakeResponse({"embeddings": [[1.0, 2.0], [1.0]]})])

    with pytest.raises(EmbeddingUnavailable, match="차원이 일정하지 않다"):
        OllamaEmbedder(session=session).embed(["가", "나"])


def test_blank_text_is_rejected() -> None:
    session = FakeSession([])

    with pytest.raises(ValueError, match="빈 문자열"):
        OllamaEmbedder(session=session).embed(["정상", "   "])

    assert session.requests == []


@pytest.mark.parametrize("batch_size,timeout", [(0, 30.0), (-1, 30.0), (64, 0), (64, -1)])
def test_invalid_parameters_are_rejected(batch_size: int, timeout: float) -> None:
    with pytest.raises(ValueError):
        OllamaEmbedder(batch_size=batch_size, timeout=timeout)


@pytest.mark.integration
def test_sample_chunks_become_vectors() -> None:
    """M1-4 완료 조건. 샘플 청크가 실제 Ollama로 벡터가 된다."""
    docs = list(JsonDocumentLoader(SAMPLES).load())
    chunks = [c for doc in docs for c in ParagraphChunker().split(doc)]
    texts = [c.text for c in chunks[:10]]

    result = OllamaEmbedder().embed(texts)

    assert len(result) == 10
    assert len({len(v) for v in result}) == 1
    # bge-m3는 1024차원이다 (MEMORY.md D-005). 차원 자체보다 모든 벡터의 차원이
    # 같은지가 중요하다. 어긋나면 색인이 거부한다.
    assert len(result[0]) == 1024
