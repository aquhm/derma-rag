"""inspect_dataset.py 테스트.

실데이터를 쓰지 않는다. 매직 바이트만 흉내 낸 가짜 파일을 tmp_path에 만든다
(MEMORY.md D-006, docs/guidelines/05-testing.md).

가장 중요한 테스트는 마지막 절이다. **보고서에 원문이 새지 않는지** 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.inspect_dataset import (
    MAX_VALUE_CHARS,
    build_report,
    classify,
    field_report,
    size_report,
)

OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32
ZIP = b"PK\x03\x04" + b"\x00" * 32
RTF = b"{\\rtf1\\ansi " + b"x" * 8


def write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


# ------------------------------------------------------------------- classify

def test_ole2_is_reported_as_legacy_binary(tmp_path: Path) -> None:
    path = write(tmp_path / "a.doc", OLE2)

    assert "OLE2" in classify(path)


def test_zip_header_is_reported_as_ooxml(tmp_path: Path) -> None:
    path = write(tmp_path / "a.doc", ZIP)

    assert "ZIP" in classify(path)


def test_rtf_is_recognized(tmp_path: Path) -> None:
    assert "RTF" in classify(write(tmp_path / "a.doc", RTF))


def test_utf8_text_is_recognized(tmp_path: Path) -> None:
    path = write(tmp_path / "a.doc", "민감성 피부".encode("utf-8"))

    assert "UTF-8" in classify(path)


def test_cp949_text_is_recognized(tmp_path: Path) -> None:
    path = write(tmp_path / "a.doc", "민감성 피부".encode("cp949"))

    assert "CP949" in classify(path)


def test_empty_file_is_recognized(tmp_path: Path) -> None:
    assert "빈 파일" in classify(write(tmp_path / "a.doc", b""))


def test_unknown_binary_reports_its_magic_bytes(tmp_path: Path) -> None:
    path = write(tmp_path / "a.doc", b"\x07\x08\x09\x0b" + b"\x00" * 16)

    result = classify(path)

    assert "알 수 없음" in result
    assert "07 08 09 0b" in result


# ------------------------------------------------------------------ size 집계

def test_size_report_counts_files_by_extension(tmp_path: Path) -> None:
    write(tmp_path / "a.doc", OLE2)
    write(tmp_path / "b.doc", OLE2)
    write(tmp_path / "c.json", b"{}")

    report = size_report([tmp_path / "a.doc", tmp_path / "b.doc", tmp_path / "c.json"])

    assert ".doc" in report
    assert "2" in report
    assert ".json" in report


# ----------------------------------------------------------------- 필드 구조

def test_field_report_lists_key_names(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text(
        json.dumps([{"doc_id": "A-1", "skin_type": "민감성", "content": "짧은 본문"}]),
        encoding="utf-8",
    )

    report = field_report([path])

    assert "doc_id" in report
    assert "skin_type" in report
    assert "content" in report


def test_field_report_shows_label_values_for_low_cardinality(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text(
        json.dumps([{"skin_type": "민감성"}, {"skin_type": "염증성"}] * 5),
        encoding="utf-8",
    )

    report = field_report([path])

    assert "민감성" in report
    assert "염증성" in report


def test_field_report_gives_length_statistics_for_long_text(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text(
        json.dumps([{"content": "가" * 400}, {"content": "나" * 600}]), encoding="utf-8"
    )

    report = field_report([path])

    assert "평균" in report
    assert "500" in report  # (400 + 600) / 2


def test_field_report_flags_reference_like_keys(tmp_path: Path) -> None:
    # QA쌍이 근거 문단 ID를 들고 있는지가 Recall@k 자동 채점의 갈림길이다.
    path = tmp_path / "qa.json"
    path.write_text(
        json.dumps([{"question": "질문", "answer": "답", "source_doc_id": "A-1"}]),
        encoding="utf-8",
    )

    report = field_report([path])

    assert "source_doc_id" in report
    assert "참조" in report


def test_field_report_handles_nested_objects(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text(
        json.dumps([{"meta": {"area": "볼"}, "qa": [{"q": "질문"}]}]), encoding="utf-8"
    )

    report = field_report([path])

    assert "meta.area" in report
    assert "qa[].q" in report


def test_broken_json_is_reported_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "k.json"
    path.write_text("{깨진", encoding="utf-8")

    assert "읽지 못함" in field_report([path])


# --------------------------------------------------- 원문 유출 방지 (S-6)

def test_long_values_never_appear_in_the_report(tmp_path: Path) -> None:
    secret = "실제문장" * 100
    path = tmp_path / "k.json"
    path.write_text(json.dumps([{"content": secret}]), encoding="utf-8")

    report = field_report([path])

    assert secret not in report
    assert secret[:MAX_VALUE_CHARS + 1] not in report


def test_high_cardinality_short_values_are_not_listed(tmp_path: Path) -> None:
    # 값 종류가 많으면 라벨 체계가 아니라 내용이다. 나열하지 않는다.
    path = tmp_path / "k.json"
    path.write_text(
        json.dumps([{"title": f"제목{i}"} for i in range(50)]), encoding="utf-8"
    )

    report = field_report([path])

    assert "제목7" not in report
    assert "고유값" in report


def test_full_report_on_a_missing_directory(tmp_path: Path) -> None:
    report = build_report(tmp_path / "없음")

    assert "없다" in report


def test_full_report_covers_every_checklist_item(tmp_path: Path) -> None:
    write(tmp_path / "raw" / "지식데이터.doc", OLE2)
    (tmp_path / "raw" / "qa.json").write_text(
        json.dumps([{"question": "질문", "answer": "답", "doc_id": "A-1"}]),
        encoding="utf-8",
    )

    report = build_report(tmp_path / "raw")

    assert "형식 판별" in report
    assert "용량" in report
    assert "필드" in report
    assert "OLE2" in report


@pytest.mark.parametrize("name", ["a.zip", "b.zip.part0"])
def test_archives_are_flagged_as_not_extracted(tmp_path: Path, name: str) -> None:
    write(tmp_path / name, ZIP)

    report = build_report(tmp_path)

    assert "압축" in report


def test_utf8_is_not_mistaken_for_cp949_when_a_char_straddles_the_cut(
    tmp_path: Path,
) -> None:
    # 헤더를 자르는 위치에서 한글 한 글자가 반으로 갈리면 utf-8 디코딩이 실패한다.
    # 그 실패로 CP949라고 판정하면 실데이터 인코딩을 잘못 기록하게 된다.
    path = write(tmp_path / "a.json", ("가" * 1000).encode("utf-8"))

    assert "UTF-8" in classify(path)


# ------------------------------------------------- 대량 파일 표본 (실데이터)

def json_file(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_field_report_groups_by_directory(tmp_path: Path) -> None:
    # 실데이터는 폴더 하나에 JSON이 수백 개 있다. 파일마다 블록을 찍으면 보고서가
    # 수천 줄이 된다.
    for i in range(5):
        json_file(tmp_path / "라벨A" / f"{i}.json", [{"doc_id": f"A-{i}"}])
    for i in range(5):
        json_file(tmp_path / "라벨B" / f"{i}.json", [{"qa_id": f"B-{i}"}])

    report = field_report(sorted(tmp_path.rglob("*.json")))

    assert report.count("라벨A") == 1
    assert report.count("라벨B") == 1


def test_field_report_samples_files_per_directory(tmp_path: Path) -> None:
    for i in range(50):
        json_file(tmp_path / "라벨" / f"{i}.json", [{"doc_id": f"A-{i}"}])

    report = field_report(sorted(tmp_path.rglob("*.json")), per_dir=3)

    assert "50개 중 3개 표본" in report


def test_sampling_still_merges_fields_from_every_sampled_file(tmp_path: Path) -> None:
    json_file(tmp_path / "라벨" / "1.json", [{"doc_id": "A-1"}])
    json_file(tmp_path / "라벨" / "2.json", [{"skin_type": "민감성"}])

    report = field_report(sorted(tmp_path.rglob("*.json")), per_dir=2)

    assert "doc_id" in report
    assert "skin_type" in report


def test_build_report_passes_the_sample_size_through(tmp_path: Path) -> None:
    for i in range(10):
        json_file(tmp_path / "라벨" / f"{i}.json", [{"doc_id": f"A-{i}"}])

    report = build_report(tmp_path, per_dir=2)

    assert "10개 중 2개 표본" in report
