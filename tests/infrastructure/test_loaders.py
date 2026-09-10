"""JsonDocumentLoader 단위 테스트.

Ollama가 필요 없으므로 integration 표시를 붙이지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.models import SkinType
from src.infrastructure.loaders import JsonDocumentLoader

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "documents.json"


def write_json(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_sample_documents() -> None:
    """M1-3 완료 조건의 앞 절반. 샘플 문서 10건이 읽힌다."""
    docs = list(JsonDocumentLoader(SAMPLES).load())

    assert len(docs) == 10


def test_maps_korean_label_to_skin_type(tmp_path: Path) -> None:
    path = write_json(tmp_path, [{"id": "d1", "text": "본문", "skin_type": "염증성"}])

    doc = next(iter(JsonDocumentLoader(path).load()))

    assert doc.skin_type is SkinType.INFLAMMATORY


def test_missing_metadata_becomes_none(tmp_path: Path) -> None:
    """원본에서 못 얻으면 None이다. 추정해서 채우지 않는다."""
    path = write_json(tmp_path, [{"id": "d1", "text": "본문"}])

    doc = next(iter(JsonDocumentLoader(path).load()))

    assert doc.skin_type is None
    assert doc.area is None


def test_unknown_skin_type_is_rejected(tmp_path: Path) -> None:
    """4대 분류에 없는 값은 데이터 오류다. 조용히 None으로 만들지 않는다."""
    path = write_json(tmp_path, [{"id": "d1", "text": "본문", "skin_type": "지성"}])

    with pytest.raises(ValueError, match="알 수 없는 skin_type"):
        list(JsonDocumentLoader(path).load())


def test_missing_required_field_is_rejected(tmp_path: Path) -> None:
    path = write_json(tmp_path, [{"id": "d1"}])

    with pytest.raises(ValueError, match="필수 필드"):
        list(JsonDocumentLoader(path).load())


def test_top_level_must_be_array(tmp_path: Path) -> None:
    path = write_json(tmp_path, {"id": "d1", "text": "본문"})

    with pytest.raises(ValueError, match="배열이 아니다"):
        list(JsonDocumentLoader(path).load())


def test_missing_file_fails_at_construction(tmp_path: Path) -> None:
    """load()가 제너레이터라 여기서 막지 않으면 첫 순회까지 오류가 미뤄진다."""
    with pytest.raises(FileNotFoundError):
        JsonDocumentLoader(tmp_path / "없는파일.json")


def test_korean_survives_round_trip(tmp_path: Path) -> None:
    """encoding="utf-8"을 빠뜨리면 Windows 기본값 CP949에서 깨진다."""
    path = write_json(tmp_path, [{"id": "d1", "text": "각질층 장벽과 피지 분비"}])

    doc = next(iter(JsonDocumentLoader(path).load()))

    assert doc.text == "각질층 장벽과 피지 분비"
