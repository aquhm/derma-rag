"""extract_dataset.py 테스트.

실데이터를 쓰지 않는다. tmp_path에 만든 가짜 zip으로만 돌린다
(MEMORY.md D-006).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

from scripts.extract_dataset import extract_all, long_path, safe_name


def make_zip(path: Path, entries: dict[str, bytes], cp949_names: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            if cp949_names:
                # 한국어 윈도우에서 만든 zip은 이름을 CP949로 넣고 UTF-8 플래그를
                # 세우지 않는다. zipfile은 그것을 CP437로 읽어 깨진 이름을 준다.
                info = zipfile.ZipInfo(name.encode("cp949").decode("cp437"))
                archive.writestr(info, payload)
            else:
                archive.writestr(name, payload)
    return path


# ------------------------------------------------------------- 이름 복원

def test_utf8_names_pass_through() -> None:
    assert safe_name("문서.json") == "문서.json"


def test_cp949_names_are_restored() -> None:
    broken = "지식데이터.json".encode("cp949").decode("cp437")

    assert safe_name(broken) == "지식데이터.json"


def test_absolute_paths_are_stripped() -> None:
    # zip slip 방지. 압축 파일이 바깥 경로를 지정할 수 있다.
    assert safe_name("/etc/passwd") == "etc/passwd"


def test_parent_traversal_is_stripped() -> None:
    assert safe_name("../../바깥.json") == "바깥.json"


def test_backslash_paths_are_normalized() -> None:
    assert safe_name("폴더\\파일.json") == "폴더/파일.json"


# --------------------------------------------------------------- 압축 해제

def test_every_zip_is_extracted(tmp_path: Path) -> None:
    make_zip(tmp_path / "a.zip", {"a.json": b"{}"})
    make_zip(tmp_path / "하위/b.zip", {"b.json": b"{}"})

    report = extract_all(tmp_path)

    assert (tmp_path / "a" / "a.json").is_file()
    assert (tmp_path / "하위" / "b" / "b.json").is_file()
    assert report.extracted == 2


def test_files_land_in_a_folder_named_after_the_zip(tmp_path: Path) -> None:
    make_zip(tmp_path / "지식데이터.zip", {"x.json": b"{}"})

    extract_all(tmp_path)

    assert (tmp_path / "지식데이터" / "x.json").is_file()


def test_cp949_entry_names_are_readable_after_extraction(tmp_path: Path) -> None:
    make_zip(tmp_path / "a.zip", {"라벨/문서.json": b"{}"}, cp949_names=True)

    extract_all(tmp_path)

    assert (tmp_path / "a" / "라벨" / "문서.json").is_file()


def test_already_extracted_zips_are_skipped(tmp_path: Path) -> None:
    make_zip(tmp_path / "a.zip", {"a.json": b"{}"})
    extract_all(tmp_path)

    report = extract_all(tmp_path)

    assert report.extracted == 0
    assert report.skipped == 1


def test_nested_zips_are_extracted_too(tmp_path: Path) -> None:
    # 압축 안에 압축이 들어 있는 경우가 있다. 한 번 더 푼다.
    inner = make_zip(tmp_path / "임시" / "inner.zip", {"deep.json": b"{}"})
    make_zip(tmp_path / "outer.zip", {"inner.zip": inner.read_bytes()})

    extract_all(tmp_path)

    assert (tmp_path / "outer" / "inner" / "deep.json").is_file()


def test_broken_zip_is_reported_not_raised(tmp_path: Path) -> None:
    (tmp_path / "깨진.zip").write_bytes(b"PK\x03\x04 not a zip")

    report = extract_all(tmp_path)

    assert report.failed == 1
    assert "깨진.zip" in report.failures[0]


def test_zip_slip_entries_stay_inside_the_target(tmp_path: Path) -> None:
    make_zip(tmp_path / "a.zip", {"../탈출.json": b"{}"})

    extract_all(tmp_path)

    assert (tmp_path / "a" / "탈출.json").is_file()
    assert not (tmp_path / "탈출.json").exists()


def test_report_counts_entries_and_bytes(tmp_path: Path) -> None:
    make_zip(tmp_path / "a.zip", {"a.json": b"0123456789", "b.json": b"012"})

    report = extract_all(tmp_path)

    assert report.entries == 2
    assert report.bytes_written == 13


def test_missing_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="경로가 없다"):
        extract_all(tmp_path / "없음")


def test_no_zip_files_is_not_an_error(tmp_path: Path) -> None:
    report = extract_all(tmp_path)

    assert report.extracted == 0
    assert report.entries == 0


# ------------------------------------------------------- 긴 경로 (Windows)

def test_long_path_prefix_is_added_on_windows() -> None:
    # 윈도우 기본 경로 한계는 260자다. AI Hub 파일 이름이 길어 이를 넘는다.
    # `\?\` 접두사를 붙이면 32,767자까지 쓸 수 있다.
    result = str(long_path(Path("C:/work/긴 이름")))

    if sys.platform == "win32":
        assert result.startswith("\\\\?\\")
        assert result.endswith("긴 이름")
    else:
        assert result == str(Path("C:/work/긴 이름"))


def test_long_path_is_absolute() -> None:
    # 상대 경로에 접두사를 붙이면 열리지 않는다.
    result = str(long_path(Path("상대/경로.json")))

    assert Path(result.replace("\\\\?\\", "")).is_absolute()


def test_long_path_is_idempotent() -> None:
    once = long_path(Path("C:/work/x"))

    assert long_path(once) == once


def test_deeply_nested_entry_is_extracted(tmp_path: Path) -> None:
    # 260자를 넘기는 경로를 실제로 만들어 본다.
    deep = "/".join(["가나다라마바사아자차카타파하" * 3] * 2) + "/문서.json"
    make_zip(tmp_path / "깊은.zip", {deep: b"{}"})

    report = extract_all(tmp_path)

    assert report.failed == 0
    assert report.entries == 1


def test_force_reextracts_an_existing_target(tmp_path: Path) -> None:
    # 중간에 실패한 압축 해제를 다시 돌리려면 필요하다. 대상 폴더가 이미 있어도
    # 건너뛰지 않는다.
    make_zip(tmp_path / "a.zip", {"a.json": b"{}"})
    extract_all(tmp_path)

    report = extract_all(tmp_path, force=True)

    assert report.extracted == 1
    assert report.skipped == 0
