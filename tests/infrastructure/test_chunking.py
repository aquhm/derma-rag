"""ParagraphChunker 단위 테스트."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.domain.models import Document, SkinType
from src.infrastructure.chunking import ParagraphChunker, WholeDocumentChunker
from src.infrastructure.loaders import JsonDocumentLoader

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "documents.json"


def test_short_document_becomes_one_chunk() -> None:
    doc = Document(id="d1", text="짧은 문단이다.")

    chunks = ParagraphChunker().split(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "짧은 문단이다."


def test_blank_line_separates_paragraphs() -> None:
    doc = Document(id="d1", text="첫 문단이다.\n\n둘째 문단이다.")

    chunks = ParagraphChunker().split(doc)

    assert [c.text for c in chunks] == ["첫 문단이다.", "둘째 문단이다."]


def test_chunk_id_carries_position() -> None:
    doc = Document(id="d1", text="첫 문단.\n\n둘째 문단.\n\n셋째 문단.")

    chunks = ParagraphChunker().split(doc)

    assert [c.id for c in chunks] == ["d1#0", "d1#1", "d1#2"]
    assert all(c.doc_id == "d1" for c in chunks)


def test_metadata_is_inherited_from_document() -> None:
    doc = Document(id="d1", text="본문.", skin_type=SkinType.PIGMENT, area="이마")

    chunk = ParagraphChunker().split(doc)[0]

    assert chunk.skin_type is SkinType.PIGMENT
    assert chunk.area == "이마"


def test_long_paragraph_splits_at_sentence_boundary() -> None:
    sentence = "각질층 장벽이 손상되면 수분 손실이 늘어난다고 기술된다."
    doc = Document(id="d1", text=" ".join([sentence] * 10))

    chunks = ParagraphChunker(max_chars=120, overlap=20).split(doc)

    assert len(chunks) > 1
    # 문장 중간에서 끊기지 않았다면 모든 청크가 문장 끝 부호로 끝난다.
    assert all(c.text.endswith(".") for c in chunks)


def test_every_chunk_respects_max_chars() -> None:
    sentence = "표면의 요철과 모공 크기 변화가 함께 나타난다고 분류된다."
    doc = Document(id="d1", text=" ".join([sentence] * 30))

    chunks = ParagraphChunker(max_chars=100, overlap=30).split(doc)

    assert all(len(c.text) <= 100 for c in chunks)


def test_overlap_repeats_tail_of_previous_chunk() -> None:
    sentence = "문맥이 경계에서 끊기는 것을 오버랩이 완화한다고 설명된다."
    doc = Document(id="d1", text=" ".join([sentence] * 8))

    chunks = ParagraphChunker(max_chars=140, overlap=20).split(doc)

    assert len(chunks) > 1
    tail = chunks[0].text[-20:]
    assert chunks[1].text.startswith(tail)


def test_zero_overlap_produces_no_repetition() -> None:
    sentence = "오버랩이 0이면 앞 청크의 꼬리가 붙지 않는다."
    doc = Document(id="d1", text=" ".join([sentence] * 6))

    chunks = ParagraphChunker(max_chars=100, overlap=0).split(doc)

    assert len(chunks) > 1
    assert not chunks[1].text.startswith(chunks[0].text[-10:])


def test_numbered_list_is_not_split_on_item_numbers() -> None:
    """(?<![0-9])가 없으면 "1. 유분을" 이 문장 경계로 잘린다."""
    text = "권장 순서는 다음과 같다. 1. 유분을 정리한다. 2. 얇게 편다."
    doc = Document(id="d1", text=text)

    chunks = ParagraphChunker(max_chars=30, overlap=5).split(doc)

    assert any("1. 유분을 정리한다." in c.text for c in chunks)


def test_single_sentence_longer_than_max_is_force_split() -> None:
    """문장 하나가 최대 길이를 넘으면 문자 수로 자른다. 조각은 원문 그대로다."""
    doc = Document(id="d1", text="가" * 300)

    chunks = ParagraphChunker(max_chars=100, overlap=20).split(doc)

    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)
    # 원문에 없던 공백이 끼어들지 않는다
    assert all(set(c.text) == {"가"} for c in chunks)
    # 오버랩 20자를 걷어내고 이어 붙이면 원문이 그대로 복원된다
    rebuilt = chunks[0].text + "".join(c.text[20:] for c in chunks[1:])
    assert rebuilt == "가" * 300


@pytest.mark.parametrize(
    "max_chars,overlap",
    [(0, 0), (-1, 0), (100, -1), (100, 100), (100, 200)],
)
def test_invalid_parameters_are_rejected(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ParagraphChunker(max_chars=max_chars, overlap=overlap)


def test_sample_documents_are_chunked() -> None:
    """M1-3 완료 조건. 샘플 문서 10건이 청크로 분할된다."""
    docs = list(JsonDocumentLoader(SAMPLES).load())
    chunker = ParagraphChunker()

    chunks = [c for doc in docs for c in chunker.split(doc)]

    assert len(docs) == 10
    assert len(chunks) > len(docs)  # 최소한 일부 문서는 여러 청크로 나뉜다
    assert all(len(c.text) <= 512 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)  # ID 충돌 없음


def test_detail_type_is_inherited_by_chunks() -> None:
    # D-028. 검색 결과는 청크 단위로 돌아오므로 청크만 보고 세부 유형을 알아야 한다.
    doc = Document(
        id="d1",
        text="첫 문단입니다.\n\n둘째 문단입니다.",
        skin_type=SkinType.INFLAMMATORY,
        area="턱",
        skin_detail="지루성 피부염",
    )

    chunks = ParagraphChunker().split(doc)

    assert len(chunks) == 2
    assert all(c.skin_detail == "지루성 피부염" for c in chunks)


# ------------------------------------- WholeDocumentChunker (M3-6, D-032)

def test_whole_document_chunker_returns_one_chunk() -> None:
    # QA쌍 색인은 질문과 답변을 한 덩어리로 넣어야 한다. 쪼개면 상위 k를 같은
    # QA의 조각들이 차지한다 (실측: 근거 3건 중 2건이 같은 QA).
    doc = Document(id="A1", text="질문: 가나다\n답변: " + "라" * 2000)

    chunks = WholeDocumentChunker().split(doc)

    assert len(chunks) == 1
    assert chunks[0].text == doc.text


def test_whole_document_chunk_id_keeps_the_convention() -> None:
    chunks = WholeDocumentChunker().split(Document(id="A1", text="본문"))

    assert chunks[0].id == "A1#0"
    assert chunks[0].doc_id == "A1"


def test_whole_document_chunker_carries_metadata() -> None:
    doc = Document(
        id="A1",
        text="본문",
        skin_type=SkinType.PIGMENT,
        area="이마",
        skin_detail="색소침착 피부",
    )

    chunk = WholeDocumentChunker().split(doc)[0]

    assert chunk.skin_type is SkinType.PIGMENT
    assert chunk.area == "이마"
    assert chunk.skin_detail == "색소침착 피부"
