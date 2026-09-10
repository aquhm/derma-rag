"""DocumentLoader 어댑터.

지금은 샘플 JSON만 읽는다. 실데이터 로더는 M2-2에서 만든다. 그때 바뀌는 것은
파일 형식과 필드명이고, 이 파일의 구조는 그대로 쓴다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from src.domain.models import Document, SkinType


class JsonDocumentLoader:
    """JSON 배열 하나를 Document 목록으로 읽는다.

    기대하는 형식은 다음과 같다. skin_type과 area는 없거나 null이어도 된다.

        [{"id": "...", "text": "...", "skin_type": "민감성", "area": "볼"}, ...]

    application.ports.DocumentLoader를 만족한다. Protocol이라 상속하지 않는다.
    """

    def __init__(self, path: Path) -> None:
        # 경로 검사를 생성 시점에 한다. load()가 제너레이터라서 여기서 막지 않으면
        # 잘못된 경로가 첫 순회 시점까지 조용히 넘어간다.
        if not path.is_file():
            raise FileNotFoundError(f"문서 파일이 없다: {path}")
        self._path = path

    def load(self) -> Iterator[Document]:
        # encoding="utf-8"을 반드시 준다. Windows 기본값은 CP949라서 한글이 깨진다
        # (docs/guidelines/00-common.md).
        raw = json.loads(self._path.read_text(encoding="utf-8"))

        if not isinstance(raw, list):
            raise ValueError(f"최상위가 배열이 아니다: {self._path}")

        # json.loads가 이미 전부 메모리에 올리므로 여기서의 제너레이터는 메모리를
        # 아껴 주지 않는다. 포트가 Iterable을 요구하는 형태를 맞추는 것이 목적이고,
        # 실제 스트리밍은 실데이터가 JSONL이면 M2-2에서 성립한다.
        for i, item in enumerate(raw):
            yield self._to_document(item, position=i)

    def _to_document(self, item: object, position: int) -> Document:
        if not isinstance(item, dict):
            raise ValueError(f"{position}번째 항목이 객체가 아니다")

        missing = [key for key in ("id", "text") if key not in item]
        if missing:
            raise ValueError(f"{position}번째 항목에 필수 필드가 없다: {missing}")

        return Document(
            id=str(item["id"]),
            text=str(item["text"]),
            skin_type=_to_skin_type(item.get("skin_type"), doc_id=str(item["id"])),
            area=item.get("area") or None,
        )


def _to_skin_type(value: object, doc_id: str) -> SkinType | None:
    """한국어 라벨을 SkinType으로 바꾼다.

    값이 없으면 None이다. 있는데 4대 분류에 없는 값이면 예외를 던진다. 추정해서
    채우지 않는다 (docs/guidelines/03-infrastructure.md 메타데이터 항목).
    """
    if value is None or value == "":
        return None
    try:
        return SkinType(value)
    except ValueError as exc:
        allowed = [s.value for s in SkinType]
        raise ValueError(
            f"알 수 없는 skin_type: {value!r} (문서 {doc_id}). 허용값: {allowed}"
        ) from exc
