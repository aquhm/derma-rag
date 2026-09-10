"""BuildIndex 유스케이스. 3단계(임베딩)와 4단계(색인 저장)를 잇는다.

배치로 나누는 이유는 두 가지다. 진행 상황이 보여야 하고(실데이터 9,377건은
오래 걸린다), 나중에 중간 저장과 재개를 붙일 자리가 여기이기 때문이다. 중간
저장은 어댑터가 아니라 이 유스케이스의 일이다
(docs/guidelines/03-infrastructure.md). 재개는 M2-4에서 붙인다.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.application.ports import Embedder, VectorIndex
from src.domain.models import Chunk

# 한 묶음의 청크 수. Embedder 어댑터도 내부에서 64건씩 요청하지만, 그것은 HTTP
# 요청 크기의 문제고 이 값은 진행 표시와 실패 단위의 문제다.
DEFAULT_BATCH_SIZE = 256

# 몇 건마다 색인을 파일로 내려쓸지. 0이면 끝에 한 번만 저장한다.
# 실데이터 재적재가 10분 걸린다. 중간에 끊기면 처음부터 다시 하게 된다.
DEFAULT_CHECKPOINT = 2000

# (처리한 건수, 전체 건수)를 받는다. 진행 표시는 interface의 일이므로 출력하지
# 않고 호출자에게 넘긴다.
ProgressCallback = Callable[[int, int], None]


class BuildIndex:
    """청크를 벡터로 만들어 색인에 넣고 파일로 저장한다."""

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size는 1 이상이어야 한다: {batch_size}")

        self._embedder = embedder
        self._index = index
        self._batch_size = batch_size

    def execute(
        self,
        chunks: list[Chunk],
        path: Path,
        on_progress: ProgressCallback | None = None,
        already_indexed: frozenset[str] = frozenset(),
        checkpoint_every: int = DEFAULT_CHECKPOINT,
    ) -> int:
        """색인에 새로 넣은 청크 수를 돌려준다.

        already_indexed에 있는 청크는 건너뛴다. 중단된 적재를 이어서 할 때
        조립부가 기존 색인의 청크 ID를 넘긴다 (M2-4).

        checkpoint_every건마다 파일로 내려쓴다. 중간 파일도 그 시점까지는 온전한
        색인이다. 청크와 벡터를 함께 저장하므로 반쯤 쓰인 상태가 생기지 않는다
        (D-018).
        """
        if not chunks:
            raise ValueError("색인할 청크가 없다")

        pending = [chunk for chunk in chunks if chunk.id not in already_indexed]
        total = len(pending)
        done = 0
        since_checkpoint = 0

        for start in range(0, total, self._batch_size):
            batch = pending[start : start + self._batch_size]
            vectors = self._embedder.embed([chunk.text for chunk in batch])
            self._index.add(batch, vectors)

            done += len(batch)
            since_checkpoint += len(batch)
            if checkpoint_every > 0 and since_checkpoint >= checkpoint_every:
                self._index.save(path)
                since_checkpoint = 0
            if on_progress is not None:
                on_progress(done, total)

        self._index.save(path)
        return total
