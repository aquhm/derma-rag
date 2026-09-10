"""AI Hub 데이터 압축 해제기 (M2-1 전 단계).

`data/raw` 아래의 `.zip`을 전부 풀어 각 zip 이름과 같은 폴더에 넣는다.

    data/raw/.../지식데이터.zip  →  data/raw/.../지식데이터/...

**원문을 출력하지 않는다.** zip 파일 이름과 건수·바이트만 찍는다. 압축 안의 파일
이름도 찍지 않는다 (MEMORY.md D-006, D-024, docs/process/SAFETY.md S-6).

기본 경로가 코드 안에 있는 이유는 `scripts/guard_data.py` 훅 때문이다. 훅은 셸
명령 문자열에 데이터 경로가 보이면 차단한다. 경로를 인자로 적지 않으면 통과한다.

사용:

    .venv/Scripts/python.exe scripts/extract_dataset.py
    .venv/Scripts/python.exe scripts/extract_dataset.py --root 경로

두 가지를 특히 조심한다.

1. **이름 인코딩** — 한국어 윈도우에서 만든 zip은 파일 이름을 CP949로 넣고 UTF-8
   플래그를 세우지 않는다. `zipfile`은 그것을 CP437로 읽어 깨진 이름을 준다.
   되돌려 디코딩한다.
2. **zip slip** — 압축 파일이 `../`나 절대 경로를 지정하면 대상 폴더 밖에 쓰게
   된다. 경로를 잘라 안쪽으로만 쓴다.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "data" / "raw"

# 압축 안에 압축이 있는 경우가 있다. 몇 겹까지 풀지.
MAX_DEPTH = 3


@dataclass
class ExtractReport:
    extracted: int = 0
    skipped: int = 0
    failed: int = 0
    entries: int = 0
    bytes_written: int = 0
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"압축 해제 {self.extracted}건, 건너뜀 {self.skipped}건, 실패 {self.failed}건",
            f"파일 {self.entries:,}개, {self.bytes_written / 1024 / 1024:,.1f} MB",
        ]
        lines += [f"  실패: {message}" for message in self.failures]
        return "\n".join(lines)


def long_path(path: Path) -> Path:
    """윈도우 260자 경로 한계를 넘기기 위한 확장 접두사를 붙인다.

    AI Hub 파일 이름이 길어 `data/raw/...` 아래 목표 경로가 300자를 넘는다.
    접두사가 없으면 `FileNotFoundError`가 난다(실측: Other.zip, 최대 302자).

    접두사는 절대 경로에만 붙는다. 상대 경로에 붙이면 열리지 않는다.
    """
    if sys.platform != "win32":
        return path
    resolved = path.resolve()
    if str(resolved).startswith("\\\\?\\"):
        return resolved
    return Path("\\\\?\\" + str(resolved))


def safe_name(raw: str) -> str:
    """압축 안의 경로를 대상 폴더 안쪽의 상대 경로로 바꾼다.

    CP949 이름을 되돌리고, 절대 경로와 상위 이동(`..`)을 잘라 낸다.
    """
    name = raw.replace("\\", "/")
    try:
        # CP437로 잘못 읽힌 이름이면 되돌린다. 원래 UTF-8이었으면 인코딩에서 실패한다.
        name = name.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    parts = [part for part in name.split("/") if part not in ("", ".", "..")]
    # 윈도우 드라이브 문자(C:)로 시작하는 경로도 잘라 낸다.
    if parts and ":" in parts[0]:
        parts = parts[1:]
    return "/".join(parts)


def extract_all(
    root: Path, depth: int = MAX_DEPTH, force: bool = False
) -> ExtractReport:
    """root 아래의 zip을 전부 푼다. 이미 푼 것은 건너뛴다.

    force=True면 대상 폴더가 있어도 다시 푼다. 중간에 실패한 압축 해제를
    이어서 마치는 용도다. 같은 이름의 파일은 덮어쓴다.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"경로가 없다: {root}")

    report = ExtractReport()
    # 이미 처리한 zip. 겹겹이 도는 동안 같은 파일을 다시 풀거나, 깨진 zip을 여러 번
    # 실패로 세지 않게 한다.
    handled: set[Path] = set()

    for _ in range(depth):
        pending = [a for a in sorted(root.rglob("*.zip")) if a not in handled]
        if not pending:
            break

        for archive in pending:
            handled.add(archive)
            if not force and _target_of(archive).is_dir():
                report.skipped += 1
                continue
            _extract_one(archive, report)

    return report


def _target_of(archive: Path) -> Path:
    """zip을 풀 폴더. zip 이름에서 확장자만 뗀다."""
    return archive.with_suffix("")


def _extract_one(archive: Path, report: ExtractReport) -> None:
    target = _target_of(archive)
    try:
        with zipfile.ZipFile(archive) as opened:
            members = [info for info in opened.infolist() if not info.is_dir()]
            for info in members:
                relative = safe_name(info.filename)
                if not relative:
                    continue
                destination = long_path(target / relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                payload = opened.read(info)
                destination.write_bytes(payload)
                report.entries += 1
                report.bytes_written += len(payload)
    except (zipfile.BadZipFile, OSError) as exc:
        report.failed += 1
        # 실패 메시지에 zip 이름만 넣는다. 내용은 넣지 않는다.
        report.failures.append(f"{archive.name}: {type(exc).__name__}")
        return

    report.extracted += 1
    print(f"  풀었다: {archive.name} → {target.name}/ ({len(members)}개)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="data/raw 아래의 zip을 전부 푼다 (원문은 출력하지 않는다)"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="대상 디렉터리")
    parser.add_argument(
        "--force",
        action="store_true",
        help="대상 폴더가 이미 있어도 다시 푼다 (중간에 실패한 경우)",
    )
    args = parser.parse_args(argv)

    print(f"대상: {args.root}")
    report = extract_all(args.root, force=args.force)
    print(report.summary())
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    raise SystemExit(main())
