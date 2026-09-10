"""지식데이터 `.doc` 원문 로더 (M3-7).

D-027에서 미룬 것을 여기서 연다. 그때는 `.doc` 형식이 섞여 있고 변환 수단이
새 의존성이라 발췌만 색인했다. 표준 라이브러리로 읽는 방법을 만들었으므로
(`src/infrastructure/doc_reader.py`) 원문을 색인 재료로 쓰는 실험이 가능해졌다.

**`사용논문` 폴더만 읽는 것이 기본이다.** 이유는 두 가지다.

1. 같은 파일이 두 곳에 있다. 최상위(`국내석사`, `해외저널` 등 3,122건)와
   `사용논문`(1,315건) + `미사용논문`(1,808건)이 같은 파일의 다른 배치다.
   전부 읽으면 문서 하나가 두 번 색인된다.
2. `미사용논문`은 QA 라벨링에 쓰이지 않은 논문이다. 답이 그 안에 없다.

문서 하나가 평균 25,608자다. 발췌(평균 170자)와 달리 청킹이 실제로 일어난다.
이것이 M3-1(청크 크기 실험)을 다시 열 수 있는 조건이다.

파일 하나가 깨져도 적재를 멈추지 않는다. 건너뛴 파일은 `errors`에 이름과 사유만
남긴다. 본문은 남기지 않는다 (SAFETY.md S-6).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

from src.domain.models import Document
from src.infrastructure.doc_reader import UnreadableDocument, read_doc

# 기본으로 읽을 폴더. QA 라벨링에 실제로 쓰인 논문만 들어 있다.
USED_PAPERS = "사용논문"

# 해시 자릿수. 3,122건 규모에서 12자리 충돌은 무시할 수 있다
# (aihub_loader.knowledge_id와 같은 기준).
ID_DIGITS = 12

# 본문이 이보다 짧으면 표지나 오류로 본다. 실측 최소가 289자였다.
MIN_TEXT_CHARS = 100


def paper_id(name: str) -> str:
    """파일 이름으로 문서 ID를 만든다.

    경로가 아니라 이름을 쓰는 이유는 같은 논문이 여러 폴더에 놓여 있기
    때문이다. 이름을 쓰면 어느 폴더에서 읽든 같은 ID가 나온다.
    """
    digest = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()
    return f"P-{digest[:ID_DIGITS]}"


class AihubKnowledgeLoader:
    """application.ports.DocumentLoader를 만족한다.

    `.doc` 파일 하나가 Document 하나다. 청킹은 다음 단계(Chunker)가 맡는다.
    """

    def __init__(
        self, root: Path, folder: str | None = USED_PAPERS, limit: int | None = None
    ) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"경로가 없다: {root}")
        if limit is not None and limit < 1:
            raise ValueError(f"limit은 1 이상이어야 한다: {limit}")
        self._root = root
        self._folder = folder
        self._limit = limit
        self.errors: list[str] = []

    def _files(self) -> list[Path]:
        paths = sorted(self._root.rglob("*.doc"))
        if self._folder is not None:
            paths = [path for path in paths if self._folder in path.parts]
        return paths

    def load(self) -> Iterator[Document]:
        files = self._files()
        if not files:
            raise ValueError(f"`.doc` 파일이 없다: {self._root} (folder={self._folder})")

        seen: set[str] = set()
        produced = 0
        for path in files:
            if self._limit is not None and produced >= self._limit:
                break

            document_id = paper_id(path.name)
            if document_id in seen:
                self.errors.append(f"{path.name}: 같은 이름을 이미 읽었다")
                continue

            try:
                text = read_doc(path)
            except UnreadableDocument as exc:
                self.errors.append(f"{path.name}: {exc}")
                continue
            except OSError as exc:
                self.errors.append(f"{path.name}: {type(exc).__name__}")
                continue

            if len(text) < MIN_TEXT_CHARS:
                self.errors.append(f"{path.name}: 본문이 {len(text)}자뿐이다")
                continue

            seen.add(document_id)
            produced += 1
            yield Document(
                id=document_id,
                text=text,
                # 논문에는 피부 유형 라벨이 없다. 폴더는 출처 종류(국내석사,
                # 해외저널 등)라서 skin_detail 자리에 넣어 화면에 보이게 한다.
                skin_detail=path.parent.name,
            )
