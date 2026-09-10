"""IngestDocuments 유스케이스. 1단계(적재)와 2단계(청크 분할)를 잇는다.

임베딩은 여기서 하지 않는다. 단계를 나누는 이유는 청킹만 바꿔 측정하기
위해서다 (M3-1, docs/guidelines/02-application.md).
"""

from __future__ import annotations

from src.application.ports import Chunker, DocumentLoader
from src.domain.models import Chunk


class IngestDocuments:
    """원본을 읽어 청크 목록으로 만든다."""

    def __init__(self, loader: DocumentLoader, chunker: Chunker) -> None:
        self._loader = loader
        self._chunker = chunker

    def execute(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        seen: set[str] = set()
        duplicated: set[str] = set()
        documents = 0

        for document in self._loader.load():
            documents += 1
            for chunk in self._chunker.split(document):
                if chunk.id in seen:
                    duplicated.add(chunk.id)
                seen.add(chunk.id)
                chunks.append(chunk)

        if documents == 0:
            # 빈 결과를 그대로 넘기면 색인 단계에서야 실패한다. 원인은 적재에 있다.
            raise ValueError("적재된 문서가 없다")
        if duplicated:
            # 색인도 중복을 막지만, 여기서 잡아야 로더나 청커를 봐야 한다는 것이
            # 드러난다.
            raise ValueError(f"청크 ID가 중복이다: {sorted(duplicated)}")

        return chunks
