"""IngestDocuments 테스트. 1~2단계(적재 → 청크 분할)를 잇는다."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from src.application.ingest import IngestDocuments
from src.domain.models import Chunk, Document, SkinType


class FakeLoader:
    """application.ports.DocumentLoader 대역."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        self.load_calls = 0

    def load(self) -> Iterable[Document]:
        self.load_calls += 1
        return iter(self._documents)


class FakeChunker:
    """문서 하나를 문장 수만큼 청크로 쪼개는 대역."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def split(self, doc: Document) -> list[Chunk]:
        self.calls.append(doc.id)
        return [
            Chunk(
                id=f"{doc.id}#{number}",
                doc_id=doc.id,
                text=part,
                skin_type=doc.skin_type,
                area=doc.area,
            )
            for number, part in enumerate(doc.text.split("|"))
        ]


def document(doc_id: str, text: str = "한 문단") -> Document:
    return Document(id=doc_id, text=text, skin_type=SkinType.SENSITIVE, area="볼")


def test_every_document_is_chunked() -> None:
    loader = FakeLoader([document("d1"), document("d2")])
    chunker = FakeChunker()

    chunks = IngestDocuments(loader=loader, chunker=chunker).execute()

    assert chunker.calls == ["d1", "d2"]
    assert [c.id for c in chunks] == ["d1#0", "d2#0"]


def test_chunks_keep_document_order() -> None:
    loader = FakeLoader([document("d1", "가|나"), document("d2", "다")])

    chunks = IngestDocuments(loader=loader, chunker=FakeChunker()).execute()

    assert [c.text for c in chunks] == ["가", "나", "다"]


def test_metadata_comes_along() -> None:
    loader = FakeLoader([document("d1")])

    chunk = IngestDocuments(loader=loader, chunker=FakeChunker()).execute()[0]

    assert chunk.skin_type is SkinType.SENSITIVE
    assert chunk.area == "볼"


def test_no_documents_is_an_error() -> None:
    # 빈 결과를 그대로 돌려주면 색인 단계에서야 실패한다. 원인은 적재에 있다.
    with pytest.raises(ValueError, match="문서가 없다"):
        IngestDocuments(loader=FakeLoader([]), chunker=FakeChunker()).execute()


def test_duplicate_chunk_ids_are_reported_here() -> None:
    # 색인도 중복을 막지만(D-018 구현), 원인은 로더나 청커에 있다. 여기서 잡아야
    # 어디를 봐야 하는지가 드러난다.
    loader = FakeLoader([document("d1"), document("d1")])

    with pytest.raises(ValueError, match="중복"):
        IngestDocuments(loader=loader, chunker=FakeChunker()).execute()
