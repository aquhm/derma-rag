"""`.doc` 본문 추출기 (M3-7).

AI Hub 지식데이터의 확장자는 전부 `.doc`이지만 실제 형식은 셋이다.

| 형식 | 건수 | 무엇인가 |
|---|---|---|
| OLE2 | 5,513 | Word 97~2003 복합 파일 |
| ZIP | 724 | 사실은 `.docx`인데 이름만 `.doc`이다 |
| RTF | 4 | 서식 있는 텍스트 |

표준 라이브러리만 쓴다. `python-docx`나 `olefile`을 넣지 않는다. 두 가지 이유다.
첫째, 새 의존성은 `docs/PACKAGES.md` 승인 대상이다. 둘째, `olefile`을 넣어도
Word 97의 조각표(piece table)는 직접 읽어야 한다. 절반만 얻고 의존성은 온전히
지는 거래다.

**파일 형식을 직접 읽는 코드다.** 규격은 MS-CFB(복합 파일)와 MS-DOC(Word 이진
형식)이다. 오프셋 상수에 무엇을 가리키는지 주석을 붙였다.
"""

from __future__ import annotations

import io
import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

# ------------------------------------------------------------------- 상수

# 복합 파일 서명. 이 8바이트로 OLE2를 판별한다.
OLE2_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# FAT 특수값. 각각 "빈 섹터"와 "사슬 끝"이다.
FREE_SECTOR = 0xFFFFFFFF
END_OF_CHAIN = 0xFFFFFFFE

# 디렉터리 항목 크기와 종류. 2=스트림, 5=루트 저장소.
DIRECTORY_ENTRY_SIZE = 128
ENTRY_STREAM = 2
ENTRY_ROOT = 5

# Word 이진 형식 서명(FIB.wIdent).
WORD_SIGNATURE = 0xA5EC

# docx의 본문 XML 이름공간.
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 본문에 남으면 안 되는 제어문자. 표 구분(0x07)과 문단 끝(0x0D)은 줄바꿈으로
# 바꾼 뒤에 이 표현식을 적용한다.
CONTROL_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f]")

# 윈도우 260자 경로 한계를 넘기는 접두사. AI Hub 파일 이름이 길어 필요하다
# (scripts/extract_dataset.py와 같은 이유).
_BACKSLASH = chr(92)
LONG_PATH_PREFIX = _BACKSLASH * 2 + "?" + _BACKSLASH


class UnreadableDocument(RuntimeError):
    """파일을 열었으나 본문을 꺼내지 못했다.

    메시지에 본문을 넣지 않는다. 파일 이름과 사유만 넣는다 (SAFETY.md S-6).
    """


def long_path(path: Path) -> Path:
    """윈도우에서만 확장 접두사를 붙인다. 절대 경로에만 붙는다."""
    if sys.platform != "win32":
        return path
    text = str(path.resolve())
    return Path(text if text.startswith(LONG_PATH_PREFIX) else LONG_PATH_PREFIX + text)


# ------------------------------------------------------- 복합 파일(OLE2) 판독

class OleCompound:
    """복합 파일 안의 스트림을 이름으로 꺼낸다.

    복합 파일은 하나의 파일 안에 든 작은 파일 시스템이다. 섹터 배열과 FAT
    (섹터 연결표)과 디렉터리로 이루어져 있다. Word 문서 본문은 그 안의
    `WordDocument` 스트림과 `1Table`(또는 `0Table`) 스트림에 나뉘어 있다.

    작은 스트림(기본 4,096바이트 미만)은 미니 FAT이라는 별도 표를 쓴다. 이것을
    빠뜨리면 짧은 문서에서만 본문이 깨진다.
    """

    def __init__(self, blob: bytes) -> None:
        if blob[:8] != OLE2_SIGNATURE:
            raise UnreadableDocument("OLE2 서명이 아니다")
        self._blob = blob
        # 헤더 0x1E: 섹터 크기 지수. 0x20: 미니 섹터 크기 지수.
        self._sector_size = 1 << struct.unpack_from("<H", blob, 0x1E)[0]
        self._mini_size = 1 << struct.unpack_from("<H", blob, 0x20)[0]
        # 헤더 0x38: 이 크기 미만이면 미니 스트림에 담긴다.
        self._cutoff = struct.unpack_from("<I", blob, 0x38)[0]
        self._root: tuple[int, int] = (0, 0)
        self._fat = self._read_fat()
        self._mini_fat = self._read_mini_fat()
        self._streams = self._read_directory()
        self._mini_stream = self._read_mini_stream()

    # ---------------------------------------------------------------- 내부

    def _sector(self, number: int) -> bytes:
        """섹터 번호로 바이트를 꺼낸다. 0번 섹터는 헤더 바로 뒤에서 시작한다."""
        start = (number + 1) * self._sector_size
        return self._blob[start : start + self._sector_size]

    @staticmethod
    def _chain(start: int, table: list[int]) -> list[int]:
        """연결표를 따라 섹터 번호를 모은다.

        이미 지난 섹터를 다시 만나면 멈춘다. 깨진 파일이 무한 반복으로 이어지면
        적재 전체가 멈춘다.
        """
        chain: list[int] = []
        seen: set[int] = set()
        current = start
        while current not in (END_OF_CHAIN, FREE_SECTOR) and current < len(table):
            if current in seen:
                break
            seen.add(current)
            chain.append(current)
            current = table[current]
        return chain

    def _difat(self) -> list[int]:
        """FAT이 어느 섹터에 있는지 적은 표. 앞 109개는 헤더 안에 있다."""
        entries = list(struct.unpack_from("<109I", self._blob, 0x4C))
        current = struct.unpack_from("<I", self._blob, 0x44)[0]
        count = struct.unpack_from("<I", self._blob, 0x48)[0]
        per_sector = self._sector_size // 4 - 1  # 마지막 칸은 다음 섹터 번호다
        for _ in range(count):
            if current in (END_OF_CHAIN, FREE_SECTOR):
                break
            block = self._sector(current)
            entries.extend(struct.unpack_from(f"<{per_sector}I", block, 0))
            current = struct.unpack_from("<I", block, per_sector * 4)[0]
        return [entry for entry in entries if entry != FREE_SECTOR]

    def _read_fat(self) -> list[int]:
        table: list[int] = []
        per_sector = self._sector_size // 4
        for number in self._difat():
            block = self._sector(number)
            if len(block) < self._sector_size:
                break
            table.extend(struct.unpack_from(f"<{per_sector}I", block, 0))
        return table

    def _read_mini_fat(self) -> list[int]:
        start = struct.unpack_from("<I", self._blob, 0x3C)[0]
        per_sector = self._sector_size // 4
        table: list[int] = []
        for number in self._chain(start, self._fat):
            table.extend(struct.unpack_from(f"<{per_sector}I", self._sector(number), 0))
        return table

    def _read_directory(self) -> dict[str, tuple[int, int]]:
        """스트림 이름 → (시작 섹터, 바이트 크기)."""
        start = struct.unpack_from("<I", self._blob, 0x30)[0]
        raw = b"".join(self._sector(number) for number in self._chain(start, self._fat))
        streams: dict[str, tuple[int, int]] = {}
        for offset in range(0, len(raw), DIRECTORY_ENTRY_SIZE):
            entry = raw[offset : offset + DIRECTORY_ENTRY_SIZE]
            if len(entry) < DIRECTORY_ENTRY_SIZE:
                break
            # 항목 64: 이름 길이(널 문자 포함 바이트 수). 66: 종류.
            name_length = struct.unpack_from("<H", entry, 64)[0]
            if name_length < 2:
                continue
            name = entry[: name_length - 2].decode("utf-16-le", errors="replace")
            kind = entry[66]
            first = struct.unpack_from("<I", entry, 116)[0]
            size = struct.unpack_from("<I", entry, 120)[0]
            if kind == ENTRY_ROOT:
                # 루트 항목은 미니 스트림 전체가 어디 있는지를 가리킨다.
                self._root = (first, size)
            elif kind == ENTRY_STREAM:
                streams[name] = (first, size)
        return streams

    def _read_mini_stream(self) -> bytes:
        first, size = self._root
        blob = b"".join(self._sector(number) for number in self._chain(first, self._fat))
        return blob[:size]

    # ---------------------------------------------------------------- 공개

    def names(self) -> list[str]:
        return list(self._streams)

    def stream(self, name: str) -> bytes:
        if name not in self._streams:
            raise UnreadableDocument(f"스트림이 없다: {name}")
        first, size = self._streams[name]
        if size < self._cutoff:
            parts = [
                self._mini_stream[
                    number * self._mini_size : (number + 1) * self._mini_size
                ]
                for number in self._chain(first, self._mini_fat)
            ]
        else:
            parts = [self._sector(number) for number in self._chain(first, self._fat)]
        return b"".join(parts)[:size]


# ------------------------------------------------------------- Word 97 본문

def word_text(compound: OleCompound) -> str:
    """조각표를 따라 본문 문자열을 잇는다.

    Word는 본문을 한 덩어리로 두지 않는다. 편집 이력 때문에 여러 조각으로 흩어져
    있고, 어느 조각이 어디에 있는지는 표(piece table)에 적혀 있다. 표를 무시하고
    스트림을 통째로 읽으면 문단 순서가 뒤섞이고 삭제된 문장이 되살아난다.

    조각마다 인코딩이 다르다. 압축 조각은 1바이트 cp1252, 나머지는 UTF-16LE다.
    한국어 문서는 대부분 UTF-16LE 조각이다.

    인자를 파일 경로가 아니라 복합 파일 객체로 받는다. 테스트에서 같은 메서드
    두 개(`names`, `stream`)만 가진 가짜를 넣어 확인할 수 있다.
    """
    document = compound.stream("WordDocument")
    if len(document) < 0x01AA:
        raise UnreadableDocument("WordDocument 스트림이 너무 짧다")
    if struct.unpack_from("<H", document, 0)[0] != WORD_SIGNATURE:
        raise UnreadableDocument("Word 97 서명이 아니다")

    # FIB 0x0A의 비트 0x0200이 어느 테이블 스트림을 쓰는지 가리킨다.
    flags = struct.unpack_from("<H", document, 0x0A)[0]
    table_name = "1Table" if flags & 0x0200 else "0Table"
    if table_name not in compound.names():
        raise UnreadableDocument(f"{table_name} 스트림이 없다")
    table = compound.stream(table_name)

    # FIB 0x01A2: 조각표(Clx)의 위치와 길이. 테이블 스트림 안의 오프셋이다.
    start, length = struct.unpack_from("<II", document, 0x01A2)
    clx = table[start : start + length]
    positions, descriptors = _piece_table(clx)

    pieces: list[str] = []
    for index, value in enumerate(descriptors):
        characters = positions[index + 1] - positions[index]
        if value & 0x40000000:  # 압축 조각: 1문자 1바이트
            offset = (value & 0x3FFFFFFF) // 2
            pieces.append(
                document[offset : offset + characters].decode("cp1252", errors="replace")
            )
        else:
            offset = value & 0x3FFFFFFF
            pieces.append(
                document[offset : offset + characters * 2].decode(
                    "utf-16-le", errors="replace"
                )
            )
    return "".join(pieces)


def _piece_table(clx: bytes) -> tuple[tuple[int, ...], list[int]]:
    """Clx에서 문자 위치 배열과 조각 서술자를 꺼낸다.

    Clx 앞에는 서식 묶음(Prc, 0x01로 시작)이 여러 개 붙어 있을 수 있다. 그것을
    건너뛰면 0x02로 시작하는 조각표가 나온다.
    """
    cursor = 0
    while cursor < len(clx) and clx[cursor] == 0x01:
        size = struct.unpack_from("<h", clx, cursor + 1)[0]
        cursor += 3 + size
    if cursor >= len(clx) or clx[cursor] != 0x02:
        raise UnreadableDocument("조각표를 찾지 못했다")

    length = struct.unpack_from("<I", clx, cursor + 1)[0]
    plc = clx[cursor + 5 : cursor + 5 + length]
    # 위치 배열은 (조각 수 + 1)개의 4바이트 값이고, 서술자는 8바이트씩이다.
    count = (len(plc) - 4) // 12
    if count < 1:
        raise UnreadableDocument("조각이 하나도 없다")
    positions = struct.unpack_from(f"<{count + 1}I", plc, 0)

    descriptors = []
    for index in range(count):
        # 서술자 8바이트 중 앞 2바이트는 플래그, 다음 4바이트가 위치다.
        base = (count + 1) * 4 + index * 8 + 2
        descriptors.append(struct.unpack_from("<I", plc, base)[0])
    return positions, descriptors


# ------------------------------------------------------------------ docx(ZIP)

def docx_text(blob: bytes) -> str:
    """`word/document.xml`의 문단을 줄바꿈으로 잇는다."""
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise UnreadableDocument(f"docx를 열지 못했다: {type(exc).__name__}") from exc

    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise UnreadableDocument("docx 본문 XML이 깨졌다") from exc

    lines = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        lines.append("".join(node.text or "" for node in paragraph.iter(f"{WORD_NS}t")))
    return "\n".join(lines)


# ------------------------------------------------------------------ 후처리

def clean_text(text: str) -> str:
    """제어문자를 걷어내고 공백을 정리한다.

    0x07은 표 칸 구분, 0x0D는 문단 끝이다. 둘 다 줄바꿈으로 바꾼다. 그대로 두면
    임베딩 입력에 보이지 않는 글자가 섞이고 청크 경계가 이상해진다.
    """
    text = text.replace("\r", "\n").replace("\x07", "\n")
    text = CONTROL_CHARS.sub(" ", text)
    text = re.sub("[ \t]+", " ", text)
    return re.sub("\n{3,}", "\n\n", text).strip()


def read_doc(path: Path) -> str:
    """확장자가 아니라 앞머리 바이트로 형식을 판별해 본문을 낸다.

    RTF와 빈 파일은 지원하지 않는다. 합쳐서 8건이라 변환 코드를 더 쓸 값이
    없다. 호출부가 `UnreadableDocument`를 잡아 건너뛴다.
    """
    blob = long_path(path).read_bytes()
    if blob[:8] == OLE2_SIGNATURE:
        return clean_text(word_text(OleCompound(blob)))
    if blob[:2] == b"PK":
        return clean_text(docx_text(blob))
    raise UnreadableDocument(f"지원하지 않는 형식이다: {path.name}")
