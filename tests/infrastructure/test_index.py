"""NumpyVectorIndex 테스트.

외부 의존이 없다. numpy 계산과 파일 저장뿐이라 전부 단위 테스트로 돌린다.
Ollama가 필요 없으므로 integration 마커를 붙이지 않는다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.domain.models import Chunk, SkinType
from src.infrastructure.index import NumpyVectorIndex


def chunk(chunk_id: str, text: str = "본문", **kwargs: object) -> Chunk:
    return Chunk(id=chunk_id, doc_id=chunk_id.split("#")[0], text=text, **kwargs)  # type: ignore[arg-type]


# 단위 벡터 셋. 서로 직교하거나 정확히 일치하도록 만들어 기대 점수를 손으로 계산할 수 있게 둔다.
X = [1.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0]
XY = [1.0, 1.0, 0.0]  # X와 45도. 정규화 후 코사인 = 0.7071...


# ---------------------------------------------------------------- add

def test_empty_index_returns_no_evidence() -> None:
    assert NumpyVectorIndex().search(X, k=5) == []


def test_search_returns_evidence_for_added_chunk() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    found = index.search(X, k=1)

    assert len(found) == 1
    assert found[0].chunk.id == "d1#0"
    assert found[0].score == pytest.approx(1.0)


def test_results_are_sorted_by_score_descending() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0"), chunk("d2#0"), chunk("d3#0")], [Y, X, XY])

    found = index.search(X, k=3)

    assert [e.chunk.id for e in found] == ["d2#0", "d3#0", "d1#0"]
    assert found[1].score == pytest.approx(1 / math.sqrt(2))
    assert found[2].score == pytest.approx(0.0)


def test_add_can_be_called_more_than_once() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])
    index.add([chunk("d2#0")], [Y])

    assert len(index) == 2
    assert {e.chunk.id for e in index.search(X, k=2)} == {"d1#0", "d2#0"}


def test_add_nothing_is_allowed() -> None:
    index = NumpyVectorIndex()
    index.add([], [])

    assert len(index) == 0


def test_add_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="개수가 다르다"):
        NumpyVectorIndex().add([chunk("d1#0")], [X, Y])


def test_add_rejects_dimension_change() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    with pytest.raises(ValueError, match="차원"):
        index.add([chunk("d2#0")], [[1.0, 0.0]])


def test_add_rejects_mixed_dimensions_in_one_call() -> None:
    with pytest.raises(ValueError, match="차원"):
        NumpyVectorIndex().add([chunk("d1#0"), chunk("d2#0")], [X, [1.0, 0.0]])


def test_add_rejects_zero_vector() -> None:
    # 길이가 0이면 정규화에서 0으로 나눈다. 여기서 막지 않으면 nan이 색인에 남는다.
    with pytest.raises(ValueError, match="영벡터"):
        NumpyVectorIndex().add([chunk("d1#0")], [[0.0, 0.0, 0.0]])


def test_add_rejects_non_finite_vector() -> None:
    with pytest.raises(ValueError, match="유한"):
        NumpyVectorIndex().add([chunk("d1#0")], [[float("nan"), 0.0, 0.0]])


def test_add_rejects_duplicate_chunk_id() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    with pytest.raises(ValueError, match="중복"):
        index.add([chunk("d1#0")], [Y])


# ---------------------------------------------------------------- search

def test_k_larger_than_index_returns_everything() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0"), chunk("d2#0")], [X, Y])

    assert len(index.search(X, k=10)) == 2


def test_search_rejects_k_below_one() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    with pytest.raises(ValueError, match="k"):
        index.search(X, k=0)


def test_search_rejects_dimension_mismatch() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    with pytest.raises(ValueError, match="차원"):
        index.search([1.0, 0.0], k=1)


def test_search_rejects_zero_query() -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    with pytest.raises(ValueError, match="영벡터"):
        index.search([0.0, 0.0, 0.0], k=1)


def test_opposite_vector_scores_minus_one_and_stays_in_range() -> None:
    # 부동소수 오차로 -1.0000001이 나오면 Evidence가 생성 시점에 터진다. 잘라서 넣는다.
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    found = index.search([-1.0, 0.0, 0.0], k=1)

    assert found[0].score == pytest.approx(-1.0)


# ---------------------------------------------------------------- save / from_file

def test_save_writes_one_file(tmp_path: Path) -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    index.save(tmp_path / "index.npz")

    assert [p.name for p in tmp_path.iterdir()] == ["index.npz"]


def test_roundtrip_keeps_search_results(tmp_path: Path) -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0"), chunk("d2#0"), chunk("d3#0")], [X, Y, XY])
    index.save(tmp_path / "index.npz")

    loaded = NumpyVectorIndex.from_file(tmp_path / "index.npz")

    before = index.search(X, k=3)
    after = loaded.search(X, k=3)
    assert [e.chunk.id for e in after] == [e.chunk.id for e in before]
    assert [e.score for e in after] == pytest.approx([e.score for e in before])


def test_roundtrip_keeps_chunk_metadata(tmp_path: Path) -> None:
    original = Chunk(
        id="d1#0",
        doc_id="d1",
        text="유분이 많은 피부는 세정 후 수분을 채운다.",
        skin_type=SkinType.INFLAMMATORY,
        area="이마",
    )
    index = NumpyVectorIndex()
    index.add([original], [X])
    index.save(tmp_path / "index.npz")

    restored = NumpyVectorIndex.from_file(tmp_path / "index.npz").search(X, k=1)[0].chunk

    assert restored == original


def test_roundtrip_keeps_missing_metadata_as_none(tmp_path: Path) -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])
    index.save(tmp_path / "index.npz")

    restored = NumpyVectorIndex.from_file(tmp_path / "index.npz").search(X, k=1)[0].chunk

    assert restored.skin_type is None
    assert restored.area is None


def test_save_creates_parent_directory(tmp_path: Path) -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])

    index.save(tmp_path / "새폴더" / "index.npz")

    assert (tmp_path / "새폴더" / "index.npz").is_file()


def test_saving_empty_index_is_rejected(tmp_path: Path) -> None:
    # 빈 색인을 저장하면 ingest가 실패한 것을 성공으로 착각한다.
    with pytest.raises(ValueError, match="비어 있"):
        NumpyVectorIndex().save(tmp_path / "index.npz")


def test_from_file_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="색인 파일"):
        NumpyVectorIndex.from_file(tmp_path / "없는파일.npz")


def test_from_file_rejects_foreign_npz(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    np.savez_compressed(path, something=np.zeros(3))

    with pytest.raises(ValueError, match="색인 형식"):
        NumpyVectorIndex.from_file(path)


def test_from_file_rejects_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "index.npz"
    np.savez_compressed(
        path,
        vectors=np.array([X, Y], dtype=np.float32),
        chunks=json.dumps([{"id": "d1#0", "doc_id": "d1", "text": "본문"}]),
    )

    with pytest.raises(ValueError, match="개수가 다르다"):
        NumpyVectorIndex.from_file(path)


def test_saved_vectors_are_normalized(tmp_path: Path) -> None:
    # 정규화를 적재 시점에 1회만 한다. 검색은 내적만 한다 (docs/guidelines/03-infrastructure.md).
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [[3.0, 4.0, 0.0]])
    index.save(tmp_path / "index.npz")

    with np.load(tmp_path / "index.npz", allow_pickle=False) as data:
        assert np.linalg.norm(data["vectors"][0]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.integration
def test_sample_chunks_are_searchable_after_save_and_load(tmp_path: Path) -> None:
    """M1-5 완료 조건. 실제 임베딩으로 색인을 만들고, 저장·로드한 뒤 검색이 동작한다."""
    from src.infrastructure.chunking import ParagraphChunker
    from src.infrastructure.embedding import OllamaEmbedder
    from src.infrastructure.loaders import JsonDocumentLoader

    samples = Path(__file__).resolve().parents[2] / "samples" / "documents.json"
    sample_chunks = [
        c for doc in JsonDocumentLoader(samples).load() for c in ParagraphChunker().split(doc)
    ]
    embedder = OllamaEmbedder()

    index = NumpyVectorIndex()
    index.add(sample_chunks, embedder.embed([c.text for c in sample_chunks]))
    index.save(tmp_path / "index.npz")
    loaded = NumpyVectorIndex.from_file(tmp_path / "index.npz")

    question = embedder.embed(["민감성 피부가 세안 후 당길 때 메이크업 전에 무엇을 주의해야 하나요?"])[0]
    found = loaded.search(question, k=3)

    assert len(index) == len(loaded) == len(sample_chunks)
    assert [e.chunk.id for e in found] == [e.chunk.id for e in index.search(question, k=3)]
    assert [e.score for e in found] == sorted((e.score for e in found), reverse=True)
    assert found[0].chunk.skin_type is SkinType.SENSITIVE


def test_roundtrip_keeps_the_detail_type(tmp_path: Path) -> None:
    # D-028. 세부 유형이 저장·로드에서 사라지면 M3-2 필터링을 붙일 수 없다.
    original = Chunk(
        id="d1#0",
        doc_id="d1",
        text="본문",
        skin_type=SkinType.SENSITIVE,
        area="볼",
        skin_detail="아토피 피부",
    )
    index = NumpyVectorIndex()
    index.add([original], [X])
    index.save(tmp_path / "index.npz")

    restored = NumpyVectorIndex.from_file(tmp_path / "index.npz").search(X, k=1)[0].chunk

    assert restored == original


def test_index_exposes_its_chunk_ids(tmp_path: Path) -> None:
    # 재개(M2-4)가 "무엇이 이미 들어갔는가"를 알아야 한다.
    index = NumpyVectorIndex()
    index.add([chunk("d1#0"), chunk("d2#0")], [X, Y])

    assert index.chunk_ids == frozenset({"d1#0", "d2#0"})


def test_chunk_ids_of_an_empty_index(tmp_path: Path) -> None:
    assert NumpyVectorIndex().chunk_ids == frozenset()


def test_chunk_ids_survive_a_roundtrip(tmp_path: Path) -> None:
    index = NumpyVectorIndex()
    index.add([chunk("d1#0")], [X])
    index.save(tmp_path / "index.npz")

    assert NumpyVectorIndex.from_file(tmp_path / "index.npz").chunk_ids == {"d1#0"}
