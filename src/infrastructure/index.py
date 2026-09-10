"""VectorIndex 어댑터.

numpy 배열 하나에 전체 벡터를 담고, 검색할 때 행렬-벡터 곱 한 번으로 점수를 낸다.
9,377 × 768이면 배열이 약 27MB(float32)라 전수 계산으로 충분하다. 벡터 DB를 쓰지
않는 이유는 MEMORY.md D-004에 있다.

코사인 유사도를 내적만으로 구하려고 적재 시점에 벡터를 L2 정규화한다. 검색마다
노름을 다시 구하면 질의 1건에 9,377번의 나눗셈이 더 붙는다
(docs/guidelines/03-infrastructure.md).

저장은 `.npz` 파일 하나다. 벡터는 `vectors` 배열에, 청크 메타데이터는 `chunks`에
JSON 문자열로 넣는다. 파일을 둘로 나누면 한쪽만 복사되거나 한쪽만 지워진 상태가
생긴다. `allow_pickle=False`로 읽으므로 npz에 임의 객체가 들어 있어도 실행되지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.models import Chunk, Evidence, SkinType

# npz 안의 키. 형식이 맞는지 확인하는 기준이기도 하다.
VECTORS_KEY = "vectors"
CHUNKS_KEY = "chunks"

# 저장 dtype. float64로 두면 파일과 메모리가 두 배가 된다. 임베딩 값의 유효자리는
# float32로 충분하고, 차이는 유사도 소수점 여섯째 자리 아래에서 난다.
DTYPE = np.float32


class NumpyVectorIndex:
    """application.ports.VectorIndex를 만족한다.

    생성자는 빈 색인을 만든다. 파일에서 읽을 때는 from_file을 쓴다. 포트에 load가
    없는 이유는 MEMORY.md D-013에 있다.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._ids: set[str] = set()
        # 아직 차원을 모르는 상태. 첫 add에서 정해진다.
        self._vectors: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunk_ids(self) -> frozenset[str]:
        """색인에 들어 있는 청크 ID.

        중단된 적재를 이어서 할 때 무엇이 이미 들어갔는지 알아야 한다 (M2-4).
        """
        return frozenset(self._ids)

    @property
    def dimension(self) -> int | None:
        """색인된 벡터의 차원. 비어 있으면 None."""
        return None if self._vectors is None else int(self._vectors.shape[1])

    # ------------------------------------------------------------------ 적재

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"청크와 벡터의 개수가 다르다: 청크 {len(chunks)}건, 벡터 {len(vectors)}건"
            )
        if not chunks:
            return

        self._reject_duplicate_ids(chunks)
        matrix = self._to_matrix(vectors)

        if self._vectors is None:
            self._vectors = matrix
        else:
            # 매 add마다 새 배열을 만든다. 적재는 배치 수만큼만 부르므로
            # (9,377건 ÷ 64건 = 147회) 복사 비용보다 코드가 단순한 쪽을 택한다.
            self._vectors = np.vstack((self._vectors, matrix))

        self._chunks.extend(chunks)
        self._ids.update(chunk.id for chunk in chunks)

    def _reject_duplicate_ids(self, chunks: list[Chunk]) -> None:
        """같은 청크 ID가 두 번 들어오는 것을 막는다.

        중복을 허용하면 검색 결과에 같은 문단이 두 번 나오고, k=5의 절반이 같은
        내용으로 채워진다. 색인을 두 번 쌓는 실수도 여기서 드러난다.
        """
        seen: set[str] = set()
        duplicated: set[str] = set()
        for chunk in chunks:
            if chunk.id in seen or chunk.id in self._ids:
                duplicated.add(chunk.id)
            seen.add(chunk.id)
        if duplicated:
            raise ValueError(f"청크 ID가 중복이다: {sorted(duplicated)}")

    def _to_matrix(self, vectors: list[list[float]]) -> np.ndarray:
        """리스트를 정규화된 (N, D) 배열로 바꾼다."""
        lengths = {len(v) for v in vectors}
        if len(lengths) != 1:
            raise ValueError(f"벡터 차원이 일정하지 않다: {sorted(lengths)}")

        expected = self.dimension
        dimension = lengths.pop()
        if expected is not None and dimension != expected:
            raise ValueError(
                f"색인의 벡터 차원과 다르다: 색인 {expected}, 입력 {dimension}"
            )

        matrix = np.asarray(vectors, dtype=DTYPE)
        if not np.isfinite(matrix).all():
            # nan이나 inf가 들어오면 점수 정렬이 조용히 망가진다. 여기서 끊는다.
            raise ValueError("벡터에 유한하지 않은 값이 있다 (nan 또는 inf)")

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError(
                "영벡터는 색인할 수 없다 (방향이 없어 유사도가 정의되지 않는다)"
            )

        return matrix / norms

    # ------------------------------------------------------------------ 검색

    def search(self, vector: list[float], k: int) -> list[Evidence]:
        if k < 1:
            raise ValueError(f"k는 1 이상이어야 한다: {k}")
        if self._vectors is None:
            return []

        query = self._to_query_vector(vector)

        # 양쪽 다 정규화돼 있으므로 내적이 곧 코사인 유사도다.
        scores = self._vectors @ query
        # 부동소수 오차로 1.0000001이 나올 수 있다. Evidence가 [-1, 1]을 강제하므로
        # 자르지 않으면 정상 검색이 예외로 끝난다.
        scores = np.clip(scores, -1.0, 1.0)

        return [
            Evidence(chunk=self._chunks[i], score=float(scores[i]))
            for i in self._top_indices(scores, k)
        ]

    def _to_query_vector(self, vector: list[float]) -> np.ndarray:
        expected = self.dimension
        if len(vector) != expected:
            raise ValueError(
                f"질의 벡터의 차원이 색인과 다르다: 색인 {expected}, 질의 {len(vector)}"
            )

        query = np.asarray(vector, dtype=DTYPE)
        if not np.isfinite(query).all():
            raise ValueError("질의 벡터에 유한하지 않은 값이 있다 (nan 또는 inf)")

        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("영벡터로는 검색할 수 없다")

        return query / norm

    @staticmethod
    def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
        """점수 상위 k개의 인덱스를 내림차순으로 준다."""
        if k >= scores.size:
            return np.argsort(-scores, kind="stable")
        # 전체 정렬은 O(N log N)이다. 상위 k개만 필요하므로 argpartition으로
        # 후보를 O(N)에 고르고 그 k개만 정렬한다.
        candidates = np.argpartition(-scores, k - 1)[:k]
        return candidates[np.argsort(-scores[candidates], kind="stable")]

    # ------------------------------------------------------------------ 저장

    def save(self, path: Path) -> None:
        if self._vectors is None:
            # 빈 색인을 저장하면 적재가 실패한 것을 성공으로 착각하게 된다.
            raise ValueError("색인이 비어 있다. 저장할 것이 없다")

        path.parent.mkdir(parents=True, exist_ok=True)
        records = json.dumps(
            [self._to_record(chunk) for chunk in self._chunks], ensure_ascii=False
        )
        np.savez_compressed(path, **{VECTORS_KEY: self._vectors, CHUNKS_KEY: records})

    @staticmethod
    def _to_record(chunk: Chunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            # Enum은 JSON에 그대로 넣을 수 없다. 원본 한국어 표기를 값으로 쓴다.
            "skin_type": None if chunk.skin_type is None else chunk.skin_type.value,
            "area": chunk.area,
            "skin_detail": chunk.skin_detail,
        }

    @classmethod
    def from_file(cls, path: Path) -> NumpyVectorIndex:
        """저장된 색인을 읽는다. 조립부(composition.py)에서만 부른다."""
        if not path.is_file():
            raise FileNotFoundError(f"색인 파일이 없다: {path}")

        # allow_pickle=False가 기본값이지만 명시한다. 색인 파일은 신뢰할 수 없는
        # 입력일 수 있고, pickle을 허용하면 읽는 것만으로 코드가 실행된다.
        with np.load(path, allow_pickle=False) as stored:
            if VECTORS_KEY not in stored or CHUNKS_KEY not in stored:
                raise ValueError(
                    f"색인 형식이 아니다. {VECTORS_KEY}와 {CHUNKS_KEY}가 필요하다: {path}"
                )
            vectors = np.asarray(stored[VECTORS_KEY], dtype=DTYPE)
            raw = str(stored[CHUNKS_KEY])

        records = cls._parse_records(raw, path)
        if vectors.ndim != 2:
            raise ValueError(f"색인 형식이 아니다. 벡터가 2차원이 아니다: {path}")
        if len(records) != vectors.shape[0]:
            raise ValueError(
                f"청크와 벡터의 개수가 다르다: 청크 {len(records)}건, "
                f"벡터 {vectors.shape[0]}건 ({path})"
            )

        index = cls()
        index._vectors = vectors
        index._chunks = [cls._to_chunk(record) for record in records]
        index._ids = {chunk.id for chunk in index._chunks}
        return index

    @staticmethod
    def _parse_records(raw: str, path: Path) -> list[dict[str, Any]]:
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"색인 형식이 아니다. 청크 JSON을 읽을 수 없다: {path}"
            ) from exc
        if not isinstance(records, list):
            raise ValueError(f"색인 형식이 아니다. 청크가 목록이 아니다: {path}")
        return records

    @staticmethod
    def _to_chunk(record: dict[str, Any]) -> Chunk:
        skin_type = record.get("skin_type")
        return Chunk(
            id=record["id"],
            doc_id=record["doc_id"],
            text=record["text"],
            # 값이 4대 분류에 없으면 SkinType이 ValueError를 낸다. 그대로 두는 것이
            # 맞다. 저장 당시와 분류 체계가 달라졌다는 뜻이다.
            skin_type=None if skin_type is None else SkinType(skin_type),
            area=record.get("area"),
            skin_detail=record.get("skin_detail"),
        )
