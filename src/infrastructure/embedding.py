"""Embedder 어댑터.

Ollama의 POST /api/embed를 쓴다. 이 엔드포인트는 여러 문장을 한 요청에 받는다.

구형 /api/embeddings는 한 요청에 한 문장만 받는다. 같은 50건으로 재보면 배치가
2.63초, 단건 루프가 104.56초로 39.7배 차이가 났다(2026-09-07, Ollama 0.33.3).
HTTP 왕복 비용이 임베딩 계산보다 크기 때문이다. 근거는 MEMORY.md D-016.
"""

from __future__ import annotations

import requests

from src.infrastructure.errors import EmbeddingUnavailable

DEFAULT_HOST = "http://localhost:11434"
# 기본 모델. 조립부(composition.py)가 실제로 쓸 모델을 지정한다 (D-005).
DEFAULT_MODEL = "bge-m3"

# 한 요청에 보낼 문장 수. 9,377건을 한 번에 보내면 타임아웃과 메모리가 걸린다.
DEFAULT_BATCH_SIZE = 64

# docs/guidelines/03-infrastructure.md 권장값. 배치 단위 요청 하나에 적용된다.
DEFAULT_TIMEOUT = 30.0


class OllamaEmbedder:
    """application.ports.Embedder를 만족한다."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size는 1 이상이어야 한다: {batch_size}")
        if timeout <= 0:
            raise ValueError(f"timeout은 0보다 커야 한다: {timeout}")

        self._model = model
        self._url = f"{host.rstrip('/')}/api/embed"
        self._batch_size = batch_size
        self._timeout = timeout
        # Session을 쓰면 TCP 연결을 재사용한다. 9,377건이면 요청이 147회라 차이가 난다.
        # 테스트에서는 가짜 세션을 넣어 Ollama 없이 동작을 확인한다.
        self._session = session if session is not None else requests.Session()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("빈 문자열은 임베딩할 수 없다")

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))

        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._session.post(
                self._url,
                json={"model": self._model, "input": batch},
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            # requests 예외가 application까지 새어 나가지 않게 여기서 바꾼다.
            # 메시지에 입력 텍스트를 넣지 않는다.
            raise EmbeddingUnavailable(
                f"임베딩 요청 실패 (model={self._model}, 건수={len(batch)}): {exc}"
            ) from exc
        except ValueError as exc:
            raise EmbeddingUnavailable(
                f"임베딩 응답이 JSON이 아니다 (model={self._model})"
            ) from exc

        return self._extract(payload, expected=len(batch))

    def _extract(self, payload: object, expected: int) -> list[list[float]]:
        """응답을 검증하고 벡터 목록을 꺼낸다.

        형식을 검사하는 이유는 Ollama 버전이나 모델이 바뀌면 응답 모양이 달라지기
        때문이다. 여기서 막지 않으면 이상한 값이 색인까지 흘러간다.
        """
        if not isinstance(payload, dict) or "embeddings" not in payload:
            raise EmbeddingUnavailable(
                f"응답에 embeddings가 없다 (model={self._model})"
            )

        vectors = payload["embeddings"]
        if not isinstance(vectors, list) or len(vectors) != expected:
            raise EmbeddingUnavailable(
                f"요청 {expected}건에 응답 벡터 수가 맞지 않는다 (model={self._model})"
            )

        dimensions = {len(v) for v in vectors}
        if len(dimensions) != 1 or 0 in dimensions:
            raise EmbeddingUnavailable(
                f"벡터 차원이 일정하지 않다: {sorted(dimensions)} (model={self._model})"
            )

        # list[float]로 맞춘다. domain과 application은 numpy를 모른다.
        return [[float(x) for x in vector] for vector in vectors]
