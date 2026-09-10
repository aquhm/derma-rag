"""BuildIndex 테스트. 3~4단계(임베딩 → 색인 저장)를 잇는다."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.application.indexing import BuildIndex
from src.domain.models import Chunk, Evidence


class FakeEmbedder:
    """application.ports.Embedder 대역. 요청 묶음을 기록한다."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(texts)
        return [[float(len(t)), 0.0] for t in texts]


class FakeIndex:
    """application.ports.VectorIndex 대역."""

    def __init__(self) -> None:
        self.added: list[tuple[list[Chunk], list[list[float]]]] = []
        self.saved: list[Path] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.added.append((chunks, vectors))

    def search(self, vector: list[float], k: int) -> list[Evidence]:
        raise AssertionError("색인 구축 경로에서 검색하지 않는다")

    def save(self, path: Path) -> None:
        self.saved.append(path)


def chunks(count: int) -> list[Chunk]:
    return [Chunk(id=f"d1#{i}", doc_id="d1", text=f"문단 {i}") for i in range(count)]


def test_all_chunks_are_embedded_and_added(tmp_path: Path) -> None:
    embedder, index = FakeEmbedder(), FakeIndex()

    count = BuildIndex(embedder=embedder, index=index).execute(
        chunks(3), tmp_path / "index.npz"
    )

    assert count == 3
    assert sum(len(c) for c, _ in index.added) == 3


def test_index_is_saved_to_the_given_path(tmp_path: Path) -> None:
    index = FakeIndex()

    BuildIndex(embedder=FakeEmbedder(), index=index).execute(
        chunks(2), tmp_path / "index.npz"
    )

    assert index.saved == [tmp_path / "index.npz"]


def test_work_is_split_into_batches(tmp_path: Path) -> None:
    # 9,377건을 한 번에 넘기면 중간 진행이 보이지 않고 실패 시 전부 잃는다.
    embedder = FakeEmbedder()

    BuildIndex(embedder=embedder, index=FakeIndex(), batch_size=2).execute(
        chunks(5), tmp_path / "index.npz"
    )

    assert [len(batch) for batch in embedder.batches] == [2, 2, 1]


def test_progress_is_reported_per_batch(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []

    BuildIndex(embedder=FakeEmbedder(), index=FakeIndex(), batch_size=2).execute(
        chunks(5), tmp_path / "index.npz", on_progress=lambda done, total: seen.append((done, total))
    )

    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_embedding_only_sends_chunk_text(tmp_path: Path) -> None:
    embedder = FakeEmbedder()

    BuildIndex(embedder=embedder, index=FakeIndex()).execute(
        chunks(2), tmp_path / "index.npz"
    )

    assert embedder.batches == [["문단 0", "문단 1"]]


def test_no_chunks_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="청크가 없다"):
        BuildIndex(embedder=FakeEmbedder(), index=FakeIndex()).execute(
            [], tmp_path / "index.npz"
        )


def test_nothing_is_saved_when_embedding_fails(tmp_path: Path) -> None:
    class BrokenEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("Ollama 없음")

    index = FakeIndex()

    with pytest.raises(RuntimeError):
        BuildIndex(embedder=BrokenEmbedder(), index=index).execute(
            chunks(2), tmp_path / "index.npz"
        )

    assert index.saved == []


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_is_rejected(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BuildIndex(embedder=FakeEmbedder(), index=FakeIndex(), batch_size=batch_size)


# ------------------------------------------------------ 중간 저장·재개 (M2-4)

def test_already_indexed_chunks_are_skipped(tmp_path: Path) -> None:
    embedder, index = FakeEmbedder(), FakeIndex()

    count = BuildIndex(embedder=embedder, index=index).execute(
        chunks(3), tmp_path / "index.npz", already_indexed=frozenset({"d1#0", "d1#1"})
    )

    assert count == 1
    assert embedder.batches == [["문단 2"]]


def test_nothing_left_to_do_is_not_an_error(tmp_path: Path) -> None:
    # 재실행했는데 전부 이미 들어가 있는 경우다. 저장만 하고 끝난다.
    index = FakeIndex()

    count = BuildIndex(embedder=FakeEmbedder(), index=index).execute(
        chunks(2), tmp_path / "index.npz", already_indexed=frozenset({"d1#0", "d1#1"})
    )

    assert count == 0
    assert index.added == []
    assert index.saved == [tmp_path / "index.npz"]


def test_checkpoint_saves_during_the_run(tmp_path: Path) -> None:
    # 9,031건 임베딩이 10분 걸린다. 중간에 끊기면 처음부터 다시 하게 된다.
    index = FakeIndex()

    BuildIndex(embedder=FakeEmbedder(), index=index, batch_size=2).execute(
        chunks(6), tmp_path / "index.npz", checkpoint_every=4
    )

    # 4건 시점에 한 번, 끝나고 한 번.
    assert len(index.saved) == 2


def test_checkpoint_can_be_turned_off(tmp_path: Path) -> None:
    index = FakeIndex()

    BuildIndex(embedder=FakeEmbedder(), index=index, batch_size=2).execute(
        chunks(6), tmp_path / "index.npz", checkpoint_every=0
    )

    assert index.saved == [tmp_path / "index.npz"]


def test_progress_counts_skipped_chunks_too(tmp_path: Path) -> None:
    # 진행 표시가 전체 대비로 보여야 어디까지 왔는지 안다.
    seen: list[tuple[int, int]] = []

    BuildIndex(embedder=FakeEmbedder(), index=FakeIndex(), batch_size=2).execute(
        chunks(5),
        tmp_path / "index.npz",
        already_indexed=frozenset({"d1#0"}),
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen[-1] == (4, 4)
