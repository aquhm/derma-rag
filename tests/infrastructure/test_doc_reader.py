"""`.doc` 본문 추출기 테스트.

실데이터를 픽스처로 쓰지 않는다 (`.claude/rules/tests.md`). 대신 복합 파일을
바이트로 직접 조립한다. 형식을 읽는 코드라서 형식을 만드는 쪽도 손으로 쓰는 것이
가장 정직하다.

가장 중요한 것은 조각표 절이다. Word는 본문을 한 덩어리로 두지 않는다. 표를
무시하고 이어 붙이면 문장이 어긋난다. 그 상황을 일부러 만들어 확인한다.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from src.infrastructure.doc_reader import (
    OLE2_SIGNATURE,
    OleCompound,
    UnreadableDocument,
    clean_text,
    docx_text,
    read_doc,
    word_text,
)

SECTOR_SIZE = 512
MINI_SIZE = 64
CUTOFF = 4096
FREE = 0xFFFFFFFF
END = 0xFFFFFFFE
FAT_SECTOR = 0xFFFFFFFD


# --------------------------------------------------------------- 복합 파일 조립

def _pad(blob: bytes, unit: int) -> bytes:
    remainder = len(blob) % unit
    return blob if remainder == 0 else blob + b"\x00" * (unit - remainder)


def _entry(name: str, kind: int, start: int, size: int) -> bytes:
    """디렉터리 항목 128바이트."""
    encoded = name.encode("utf-16-le") + b"\x00\x00"
    entry = bytearray(128)
    entry[: len(encoded)] = encoded
    struct.pack_into("<H", entry, 64, len(encoded))
    entry[66] = kind
    entry[67] = 1  # 색 플래그. 읽는 쪽은 보지 않지만 규격상 채운다
    struct.pack_into("<III", entry, 68, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
    struct.pack_into("<I", entry, 116, start)
    struct.pack_into("<I", entry, 120, size)
    return bytes(entry)


def build_compound(streams: dict[str, bytes]) -> bytes:
    """이름 → 내용 사전으로 최소 복합 파일을 만든다.

    4,096바이트 미만 스트림은 미니 스트림에 넣는다. 실제 Word 문서도 그렇게
    저장하며, 이 경로를 빠뜨리면 짧은 테이블 스트림에서만 깨진다.
    """
    small = {name: blob for name, blob in streams.items() if len(blob) < CUTOFF}
    large = {name: blob for name, blob in streams.items() if len(blob) >= CUTOFF}

    # 미니 스트림: 작은 스트림을 64바이트 단위로 이어 붙인다.
    mini_blob = b""
    mini_start: dict[str, int] = {}
    for name, blob in small.items():
        mini_start[name] = len(mini_blob) // MINI_SIZE
        mini_blob += _pad(blob, MINI_SIZE) if blob else b"\x00" * MINI_SIZE

    mini_sectors = max(1, len(_pad(mini_blob, SECTOR_SIZE)) // SECTOR_SIZE)
    mini_first = 3  # 0=FAT, 1=디렉터리, 2=미니 FAT
    next_sector = mini_first + mini_sectors

    large_first: dict[str, int] = {}
    large_counts: dict[str, int] = {}
    for name, blob in large.items():
        count = len(_pad(blob, SECTOR_SIZE)) // SECTOR_SIZE
        large_first[name] = next_sector
        large_counts[name] = count
        next_sector += count

    # FAT: 섹터 연결표.
    fat = [FREE] * (SECTOR_SIZE // 4)
    fat[0] = FAT_SECTOR
    fat[1] = END
    fat[2] = END
    for index in range(mini_sectors):
        sector = mini_first + index
        fat[sector] = END if index == mini_sectors - 1 else sector + 1
    for name, count in large_counts.items():
        for index in range(count):
            sector = large_first[name] + index
            fat[sector] = END if index == count - 1 else sector + 1

    # 미니 FAT: 미니 섹터 연결표.
    mini_fat = [FREE] * (SECTOR_SIZE // 4)
    for name, blob in small.items():
        count = max(1, len(_pad(blob, MINI_SIZE)) // MINI_SIZE)
        for index in range(count):
            slot = mini_start[name] + index
            mini_fat[slot] = END if index == count - 1 else slot + 1

    directory = _entry("Root Entry", 5, mini_first, len(mini_blob))
    for name, blob in streams.items():
        start = mini_start[name] if name in small else large_first[name]
        directory += _entry(name, 2, start, len(blob))
    directory = _pad(directory, SECTOR_SIZE)

    header = bytearray(SECTOR_SIZE)
    header[:8] = OLE2_SIGNATURE
    struct.pack_into("<H", header, 0x1A, 3)  # 주 버전
    struct.pack_into("<H", header, 0x1C, 0xFFFE)  # 리틀 엔디언
    struct.pack_into("<H", header, 0x1E, 9)  # 섹터 512바이트
    struct.pack_into("<H", header, 0x20, 6)  # 미니 섹터 64바이트
    struct.pack_into("<I", header, 0x2C, 1)  # FAT 섹터 수
    struct.pack_into("<I", header, 0x30, 1)  # 디렉터리 첫 섹터
    struct.pack_into("<I", header, 0x38, CUTOFF)
    struct.pack_into("<I", header, 0x3C, 2)  # 미니 FAT 첫 섹터
    struct.pack_into("<I", header, 0x40, 1)  # 미니 FAT 섹터 수
    struct.pack_into("<I", header, 0x44, END)  # DIFAT 첫 섹터
    struct.pack_into("<I", header, 0x48, 0)
    struct.pack_into("<109I", header, 0x4C, *([0] + [FREE] * 108))

    body = b"".join(
        [
            struct.pack(f"<{len(fat)}I", *fat),
            directory,
            struct.pack(f"<{len(mini_fat)}I", *mini_fat),
            _pad(mini_blob, SECTOR_SIZE) if mini_blob else b"\x00" * SECTOR_SIZE,
        ]
    )
    for name, blob in large.items():
        body += _pad(blob, SECTOR_SIZE)
    return bytes(header) + body


# ------------------------------------------------------------ Word 문서 조립

def build_word(
    pieces: list[tuple[str, bool]], use_first_table: bool = True
) -> tuple[bytes, bytes]:
    """조각 목록으로 (WordDocument, Clx) 한 쌍을 만든다.

    pieces는 (본문, 압축 여부)다. 압축 조각은 1바이트 인코딩(cp1252)이다.
    """
    document = bytearray(b"\x00" * 0x1000)
    struct.pack_into("<H", document, 0, 0xA5EC)
    struct.pack_into("<H", document, 0x0A, 0x0200 if use_first_table else 0x0000)

    offsets: list[int] = []
    cursor = 0x800
    for text, compressed in pieces:
        blob = text.encode("cp1252") if compressed else text.encode("utf-16-le")
        document[cursor : cursor + len(blob)] = blob
        offsets.append(cursor)
        cursor += len(blob) + 16  # 조각 사이에 빈 곳을 둔다

    positions = [0]
    for text, _ in pieces:
        positions.append(positions[-1] + len(text))

    plc = struct.pack(f"<{len(positions)}I", *positions)
    for index, (_, compressed) in enumerate(pieces):
        value = offsets[index] * 2 | 0x40000000 if compressed else offsets[index]
        plc += struct.pack("<HIH", 0, value, 0)

    clx = b"\x02" + struct.pack("<I", len(plc)) + plc
    struct.pack_into("<II", document, 0x01A2, 0, len(clx))
    return bytes(document), clx


def word_compound(pieces: list[tuple[str, bool]], table: str = "1Table") -> OleCompound:
    document, clx = build_word(pieces, use_first_table=table == "1Table")
    return OleCompound(build_compound({"WordDocument": document, table: clx}))


# ----------------------------------------------------------- 복합 파일 판독

def test_stream_names_are_listed() -> None:
    compound = OleCompound(build_compound({"WordDocument": b"x" * 5000, "1Table": b"t"}))

    assert set(compound.names()) == {"WordDocument", "1Table"}


def test_large_stream_is_read_from_the_main_fat() -> None:
    payload = bytes(range(256)) * 24  # 6,144바이트. 경계값 4,096 초과
    compound = OleCompound(build_compound({"WordDocument": payload}))

    assert compound.stream("WordDocument") == payload


def test_small_stream_is_read_from_the_mini_fat() -> None:
    # 4,096바이트 미만은 미니 스트림에 담긴다. 이 경로를 빠뜨리면 짧은 테이블
    # 스트림만 조용히 깨진다.
    payload = bytes(range(200))
    compound = OleCompound(build_compound({"1Table": payload}))

    assert compound.stream("1Table") == payload


def test_stream_spanning_several_mini_sectors() -> None:
    payload = bytes(range(256)) * 4  # 1,024바이트 = 미니 섹터 16개
    compound = OleCompound(build_compound({"1Table": payload}))

    assert compound.stream("1Table") == payload


def test_missing_stream_is_reported() -> None:
    compound = OleCompound(build_compound({"1Table": b"t"}))

    with pytest.raises(UnreadableDocument, match="스트림이 없다"):
        compound.stream("WordDocument")


def test_non_ole2_bytes_are_rejected() -> None:
    with pytest.raises(UnreadableDocument, match="OLE2"):
        OleCompound(b"not a compound file")


# ---------------------------------------------------------------- 조각표

def test_pieces_are_joined_in_table_order() -> None:
    compound = word_compound([("피부 장벽을 ", False), ("먼저 살핀다", False)])

    assert word_text(compound) == "피부 장벽을 먼저 살핀다"


def test_pieces_of_different_lengths_keep_their_boundaries() -> None:
    compound = word_compound([("가나다라마바사", False), ("아자", False)])

    assert word_text(compound) == "가나다라마바사아자"


def test_compressed_piece_uses_single_byte_encoding() -> None:
    compound = word_compound([("Skin barrier", True)])

    assert word_text(compound) == "Skin barrier"


def test_mixed_compressed_and_wide_pieces() -> None:
    compound = word_compound([("SPF ", True), ("차단제", False)])

    assert word_text(compound) == "SPF 차단제"


def test_zero_table_stream_is_used_when_the_flag_is_off() -> None:
    compound = word_compound([("본문", False)], table="0Table")

    assert word_text(compound) == "본문"


def test_missing_table_stream_is_reported() -> None:
    document, _ = build_word([("본문", False)])
    compound = OleCompound(build_compound({"WordDocument": document}))

    with pytest.raises(UnreadableDocument, match="1Table"):
        word_text(compound)


def test_document_without_the_word_signature_is_rejected() -> None:
    compound = OleCompound(build_compound({"WordDocument": b"\x00" * 0x1000}))

    with pytest.raises(UnreadableDocument, match="서명"):
        word_text(compound)


def test_truncated_document_is_rejected() -> None:
    short = bytearray(b"\x00" * 0x100)
    struct.pack_into("<H", short, 0, 0xA5EC)
    compound = OleCompound(build_compound({"WordDocument": bytes(short)}))

    with pytest.raises(UnreadableDocument, match="짧다"):
        word_text(compound)


# ------------------------------------------------------------------- docx

def _docx(paragraphs: list[str]) -> bytes:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    xml = f'<w:document xmlns:w="{namespace}"><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def test_docx_paragraphs_become_lines() -> None:
    assert docx_text(_docx(["첫 문단", "둘째 문단"])) == "첫 문단\n둘째 문단"


def test_docx_joins_split_runs_inside_one_paragraph() -> None:
    # 워드는 한 문장을 여러 run으로 쪼개 저장한다. 그대로 두면 단어가 끊긴다.
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    xml = (
        f'<w:document xmlns:w="{namespace}"><w:body><w:p>'
        "<w:r><w:t>각질</w:t></w:r><w:r><w:t>층</w:t></w:r>"
        "</w:p></w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)

    assert docx_text(buffer.getvalue()) == "각질층"


def test_zip_without_document_xml_is_reported() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("other.txt", "x")

    with pytest.raises(UnreadableDocument, match="docx"):
        docx_text(buffer.getvalue())


# ------------------------------------------------------------------- 후처리

def test_paragraph_marks_and_cell_marks_become_newlines() -> None:
    assert clean_text("첫 줄\r둘째 줄\x07셋째 줄") == "첫 줄\n둘째 줄\n셋째 줄"


def test_control_characters_are_removed() -> None:
    # 0x13, 0x14, 0x15는 필드 표시다. 화면에 보이지 않지만 임베딩에는 들어간다.
    assert clean_text("자외선\x13\x14차단\x15") == "자외선 차단"


def test_runs_of_blank_lines_collapse() -> None:
    assert clean_text("가\n\n\n\n나") == "가\n\n나"


def test_leading_and_trailing_space_is_dropped() -> None:
    assert clean_text("  \n 본문 \n  ") == "본문"


# ---------------------------------------------------------------- 형식 판별

def test_read_doc_handles_an_ole2_file(tmp_path: Path) -> None:
    document, clx = build_word([("보습제를 먼저 바른다", False)])
    path = tmp_path / "논문.doc"
    path.write_bytes(build_compound({"WordDocument": document, "1Table": clx}))

    assert read_doc(path) == "보습제를 먼저 바른다"


def test_read_doc_handles_a_zip_disguised_as_doc(tmp_path: Path) -> None:
    path = tmp_path / "논문.doc"
    path.write_bytes(_docx(["제목", "본문입니다"]))

    assert read_doc(path) == "제목\n본문입니다"


def test_rtf_is_not_supported(tmp_path: Path) -> None:
    # 4건뿐이라 변환 코드를 쓰지 않기로 했다. 조용히 빈 문자열을 내면 안 된다.
    path = tmp_path / "문서.doc"
    path.write_bytes(b"{\\rtf1\\ansi text}")

    with pytest.raises(UnreadableDocument, match="지원하지 않는"):
        read_doc(path)


def test_empty_file_is_not_supported(tmp_path: Path) -> None:
    path = tmp_path / "빈.doc"
    path.write_bytes(b"")

    with pytest.raises(UnreadableDocument, match="지원하지 않는"):
        read_doc(path)
