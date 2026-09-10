"""AI Hub 원본 데이터 검사기 (M2-1).

`docs/DATASETS.md`의 "수령 후 확인할 것" 다섯 항목에 한 번에 답하는 것이 목적이다.

1. `.doc`이 구형 바이너리인가 `.docx`인가  → 매직 바이트로 판별
2. 지식데이터 1건의 평균 길이              → 문자열 필드 길이 통계
3. QA쌍이 근거 문단 ID를 포함하는가        → 참조로 보이는 키 이름 표시
4. 메타데이터 필드명과 값 체계             → 키 경로와 저빈도 라벨 값
5. 실제 전체 용량                          → 확장자별 파일 수와 바이트

**원문을 출력하지 않는다.** 이 규칙이 이 스크립트의 설계 제약이다
(MEMORY.md D-006, docs/process/SAFETY.md S-6).

- 문자열 값은 30자를 넘으면 길이 통계만 낸다.
- 30자 이하라도 고유값이 20종을 넘으면 라벨이 아니라 내용으로 보고 나열하지 않는다.
- 파일 내용을 다른 곳에 복사하거나 저장하지 않는다.

사용:

    .venv/Scripts/python.exe scripts/inspect_dataset.py
    .venv/Scripts/python.exe scripts/inspect_dataset.py --root 경로 --out 파일

기본 경로가 코드 안에 있는 이유는 `scripts/guard_data.py` 훅 때문이다. 훅은 셸
명령 문자열에 데이터 경로가 보이면 차단한다. 경로를 인자로 적지 않으면 통과한다.
훅이 막으려는 것은 원문이 대화와 로그로 새는 것이고, 이 스크립트는 집계값만
출력하므로 그 목적에 어긋나지 않는다.
"""

from __future__ import annotations

import argparse
import codecs
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw"

# 이 길이를 넘는 문자열 값은 본문으로 본다. 값을 찍지 않고 길이만 센다.
MAX_VALUE_CHARS = 30

# 고유값이 이보다 많으면 라벨 체계가 아니다. 나열하지 않는다.
MAX_DISTINCT = 20

# JSON 하나에서 훑을 최대 레코드 수. 9,377건 전부를 훑을 필요가 없다.
SAMPLE_LIMIT = 2000

# 폴더 하나에서 열어 볼 JSON 수. 같은 폴더의 파일은 형식이 같다.
DEFAULT_PER_DIR = 3

# 앞부분 몇 바이트만 읽는다. 판별에 그 이상은 필요 없다.
HEADER_BYTES = 512

SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "구형 바이너리 (OLE2 복합 문서)"),
    (b"PK\x03\x04", "ZIP 컨테이너 (OOXML .docx 또는 압축 파일)"),
    (b"{\\rtf", "RTF 서식 문서"),
    (b"%PDF-", "PDF"),
    (b"HWP Document File", "HWP 3.0 문서"),
)

ARCHIVE_SUFFIXES = (".zip", ".gz", ".tar", ".7z", ".rar")

# 다른 레코드를 가리키는 것으로 보이는 키 이름. QA쌍이 근거 문단을 지목하는지가
# Recall@k 자동 채점 가능 여부를 가른다 (docs/DATASETS.md).
REFERENCE_HINTS = ("id", "idx", "no", "doc", "source", "ref", "passage", "context")


def classify(path: Path) -> str:
    """파일 형식을 매직 바이트로 판별한다. 내용은 돌려주지 않는다."""
    header = path.read_bytes()[:HEADER_BYTES]
    if not header:
        return "빈 파일"

    for signature, name in SIGNATURES:
        if header.startswith(signature):
            return name

    if b"\x00" not in header:
        for encoding, label in (("utf-8", "텍스트 (UTF-8)"), ("cp949", "텍스트 (CP949 추정)")):
            if _decodes(header, encoding):
                return label

    magic = " ".join(f"{byte:02x}" for byte in header[:4])
    return f"알 수 없음 (매직 바이트 {magic})"


def _decodes(data: bytes, encoding: str) -> bool:
    """앞부분만 잘라 온 바이트가 이 인코딩으로 읽히는가.

    증분 디코더를 쓰는 이유는 512바이트에서 자르면 한글 한 글자가 반으로 갈리기
    때문이다. 일반 decode()는 그것을 오류로 보고, UTF-8 파일이 CP949나 "알 수
    없음"으로 잘못 판정된다.
    """
    decoder = codecs.getincrementaldecoder(encoding)()
    try:
        decoder.decode(data, False)
    except UnicodeDecodeError:
        return False
    return True


def scan_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def size_report(paths: list[Path]) -> str:
    """확장자별 파일 수와 용량."""
    if not paths:
        return "파일이 없다"

    counts: dict[str, int] = defaultdict(int)
    total_bytes: dict[str, int] = defaultdict(int)
    for path in paths:
        suffix = path.suffix.lower() or "(확장자 없음)"
        counts[suffix] += 1
        total_bytes[suffix] += path.stat().st_size

    lines = [f"전체 {len(paths)}개 파일, {_bytes(sum(total_bytes.values()))}", ""]
    lines += [
        f"  {suffix:<16} {counts[suffix]:>6}개  {_bytes(total_bytes[suffix])}"
        for suffix in sorted(counts, key=lambda s: -total_bytes[s])
    ]
    return "\n".join(lines)


def field_report(paths: list[Path], per_dir: int = DEFAULT_PER_DIR) -> str:
    """JSON 파일의 키 경로, 타입, 길이 통계, 라벨 값.

    폴더 단위로 묶고 폴더마다 최대 per_dir개만 읽는다. 실데이터는 폴더 하나에
    JSON이 수백 개 있고, 전부 읽으면 보고서가 수천 줄이 된다. 같은 폴더의 파일은
    형식이 같으므로 몇 개만 봐도 구조가 드러난다.
    """
    json_paths = [p for p in paths if p.suffix.lower() == ".json"]
    if not json_paths:
        return "JSON 파일이 없다"

    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in json_paths:
        grouped[path.parent].append(path)

    blocks: list[str] = []
    for directory in sorted(grouped):
        files = sorted(grouped[directory])
        sampled = files[:per_dir] if per_dir > 0 else files
        blocks.append(
            f"[{directory.name or directory}] {len(files)}개 중 {len(sampled)}개 표본"
        )

        fields: dict[str, _Field] = {}
        shapes: set[str] = set()
        for path in sampled:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                blocks.append(f"  읽지 못함: {path.name} ({type(exc).__name__})")
                continue
            shapes.add(_shape(payload))
            _collect_into(_records(payload), fields)

        if shapes:
            blocks.append(f"  최상위: {' / '.join(sorted(shapes))}")
        if not fields:
            blocks.append("  필드를 찾지 못했다")
        blocks += [f"  {line}" for line in _describe(fields)]
        blocks.append("")

    return "\n".join(blocks).rstrip()


def build_report(root: Path, per_dir: int = DEFAULT_PER_DIR) -> str:
    """다섯 항목을 담은 보고서 전체."""
    lines = ["# 데이터셋 검사 보고서", "", f"대상 경로: {root}", ""]

    if not root.is_dir():
        lines.append(f"경로가 없다: {root}")
        lines.append("데이터를 먼저 내려받아 배치하십시오 (docs/DATASETS.md 수령 절차).")
        return "\n".join(lines)

    paths = scan_files(root)

    lines += ["## 1. 용량", "", size_report(paths), ""]
    lines += ["## 2. 형식 판별", "", _format_report(paths), ""]
    lines += ["## 3. 필드 구조", "", field_report(paths, per_dir=per_dir), ""]

    archives = [p for p in paths if _is_archive(p)]
    if archives:
        lines += [
            "## 4. 압축 파일",
            "",
            f"압축 파일 {len(archives)}개가 남아 있다. 풀기 전에는 형식과 필드를 볼 수 없다.",
            "분할 파일 병합 방법은 docs/DATASETS.md에 있다.",
            "",
        ]

    return "\n".join(lines).rstrip() + "\n"


# ----------------------------------------------------------------- 내부 함수


def _format_report(paths: list[Path]) -> str:
    if not paths:
        return "파일이 없다"

    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for path in paths:
        suffix = path.suffix.lower() or "(확장자 없음)"
        grouped[(suffix, classify(path))] += 1

    lines = []
    for (suffix, kind), count in sorted(grouped.items()):
        lines.append(f"  {suffix:<16} {count:>6}개  {kind}")

    if any(suffix == ".doc" and "OLE2" in kind for suffix, kind in grouped):
        lines += [
            "",
            "  .doc이 OLE2 구형 바이너리다. python-docx로 열리지 않는다.",
            "  변환이 필요하다 (LibreOffice CLI 등). 근거는 ERRORS.md의 데이터 형식 항목.",
        ]
    return "\n".join(lines)


def _records(payload: Any) -> list[dict[str, Any]]:
    """레코드 목록을 꺼낸다. 최상위가 dict면 가장 긴 리스트 값을 쓴다."""
    if isinstance(payload, list):
        return [r for r in payload[:SAMPLE_LIMIT] if isinstance(r, dict)]
    if isinstance(payload, dict):
        lists = [v for v in payload.values() if isinstance(v, list)]
        if lists:
            longest = max(lists, key=len)
            return [r for r in longest[:SAMPLE_LIMIT] if isinstance(r, dict)]
        return [payload]
    return []


def _shape(payload: Any) -> str:
    if isinstance(payload, list):
        return f"리스트 {len(payload)}건"
    if isinstance(payload, dict):
        return f"사전 (키 {len(payload)}개: {', '.join(list(payload)[:8])})"
    return type(payload).__name__


class _Field:
    """키 경로 하나의 관찰 결과. 값 자체는 짧은 것만 들고 있는다."""

    def __init__(self) -> None:
        self.count = 0
        self.types: set[str] = set()
        self.lengths: list[int] = []
        self.short_values: set[str] = set()
        self.too_many = False

    def observe(self, value: Any) -> None:
        self.count += 1
        self.types.add(type(value).__name__)
        if not isinstance(value, str):
            return
        self.lengths.append(len(value))
        if len(value) > MAX_VALUE_CHARS:
            # 본문이다. 값을 들고 있지 않는다.
            self.too_many = True
            return
        if not self.too_many:
            self.short_values.add(value)
            if len(self.short_values) > MAX_DISTINCT:
                self.short_values.clear()
                self.too_many = True


def _collect_into(records: list[dict[str, Any]], fields: dict[str, _Field]) -> None:
    """여러 파일의 관찰 결과를 같은 사전에 누적한다."""

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{prefix}.{key}" if prefix else str(key))
            return
        if isinstance(value, list):
            for item in value[:50]:
                walk(item, f"{prefix}[]")
            return
        if prefix:
            fields.setdefault(prefix, _Field()).observe(value)

    for record in records:
        walk(record, "")


def _describe(fields: dict[str, _Field]) -> list[str]:
    lines = []
    for name in sorted(fields):
        field = fields[name]
        parts = [f"{name} ({'/'.join(sorted(field.types))}, {field.count}건)"]

        if field.lengths:
            parts.append(
                f"길이 최소 {min(field.lengths)} / 평균 "
                f"{statistics.mean(field.lengths):.1f} / 최대 {max(field.lengths)}자"
            )
        if field.short_values and not field.too_many:
            parts.append(f"값: {', '.join(sorted(field.short_values))}")
        elif field.too_many:
            unique = len({length for length in field.lengths})
            parts.append(f"고유값 다수 (길이 종류 {unique}종). 값은 찍지 않는다")
        if _looks_like_reference(name):
            parts.append("← 참조 후보")

        lines.append("  ".join(parts))
    return lines


def _looks_like_reference(name: str) -> bool:
    tail = name.split(".")[-1].replace("[]", "").lower()
    return any(hint in tail for hint in REFERENCE_HINTS)


def _is_archive(path: Path) -> bool:
    lowered = path.name.lower()
    return lowered.endswith(ARCHIVE_SUFFIXES) or ".zip.part" in lowered


def _bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Hub 원본 데이터의 형식·용량·필드 구조를 집계한다 (원문은 출력하지 않는다)"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="검사할 디렉터리")
    parser.add_argument("--out", type=Path, default=None, help="보고서를 저장할 파일")
    parser.add_argument(
        "--per-dir",
        type=int,
        default=DEFAULT_PER_DIR,
        help=f"폴더마다 열어 볼 JSON 수. 0이면 전부 (기본값: {DEFAULT_PER_DIR})",
    )
    args = parser.parse_args(argv)

    report = build_report(args.root, per_dir=args.per_dir)
    print(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"저장: {args.out}")

    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    raise SystemExit(main())
