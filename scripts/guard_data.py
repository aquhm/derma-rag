"""PreToolUse 훅. data/ 아래 원본 데이터 영역에 대한 접근을 차단한다.

AI Hub 이용약관이 원본 데이터의 외부 공개·공유·재배포를 금지한다.
저장소에 섞여 들어가는 경로 자체를 막는 것이 목적이다.
근거: MEMORY.md D-006, docs/DATASETS.md, docs/process/SAFETY.md S-6

검사 대상은 두 가지다.

- ``Write`` / ``Edit``: ``tool_input.file_path``가 data/ 아래면 차단한다.
- ``Bash``: ``tool_input.command`` 안에 data/ 아래 경로가 있으면 차단한다.
  단, 파일 목록만 보는 명령(ls, find, du, stat, tree)은 통과시킨다.
  내용을 읽는 명령(cat, head, grep 등)과 복사·이동 명령은 막는다.

``data/README.md``와 ``data/`` 디렉터리 자체는 예외로 허용한다.

사용 (settings.json):
    python scripts/guard_data.py

종료 코드는 항상 0이다. 차단은 stdout JSON으로 전달한다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ALLOWED = {(DATA_DIR / "README.md").resolve()}

# 파일 이름만 보여 주는 명령. 내용을 읽지 않으므로 통과시킨다.
LISTING_VERBS = {"ls", "dir", "find", "tree", "du", "stat"}

# find가 다른 명령을 대신 실행하거나 파일을 지우는 형태는 통과시키지 않는다.
UNSAFE_LISTING_FLAGS = {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint"}

# 파이프와 연결 연산자로 명령을 쪼갠다. 각 조각을 따로 판정한다.
SEGMENT_SPLIT = re.compile(r"\|\||&&|[;|\n]")

REASON_WRITE = (
    "data/ 아래는 AI Hub 원본 데이터 영역이라 쓰기가 차단됩니다. "
    "원본과 가공본 모두 저장소에 남기지 않습니다 "
    "(MEMORY.md D-006, docs/process/SAFETY.md S-6). "
    "데이터는 aihubshell로 받아 직접 배치하십시오."
)

REASON_BASH = (
    "data/ 아래는 AI Hub 원본 데이터 영역이라 셸에서의 접근이 차단됩니다. "
    "원본 내용을 대화나 로그로 옮기지 않습니다 "
    "(MEMORY.md D-006, docs/process/SAFETY.md S-6). "
    "파일 목록 확인은 ls, find, du, stat, tree로 하십시오."
)


def target_path(payload: dict) -> Path | None:
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not raw:
        return None
    try:
        return Path(str(raw)).resolve()
    except (OSError, ValueError):
        return None


def is_blocked(path: Path) -> bool:
    if path in ALLOWED:
        return False
    try:
        path.relative_to(DATA_DIR)
    except ValueError:
        return False
    return True


def resolve_token(token: str) -> Path | None:
    """셸 토큰을 경로로 해석한다. 경로가 아니면 None."""
    cleaned = token.strip("\"'").rstrip("/\\")
    if not cleaned or cleaned.startswith("-"):
        return None
    try:
        candidate = Path(cleaned)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate.resolve()
    except (OSError, ValueError):
        return None


def touches_data(path: Path) -> bool:
    """data/ 디렉터리 자체 또는 그 아래를 가리키는지."""
    if path in ALLOWED:
        return False
    if path == DATA_DIR:
        return True
    try:
        path.relative_to(DATA_DIR)
    except ValueError:
        return False
    return True


def inside_data(path: Path) -> bool:
    """data/ 아래의 파일이나 하위 디렉터리를 가리키는지. data/ 자체는 제외."""
    return path != DATA_DIR and touches_data(path)


def blocked_bash(command: str) -> bool:
    for segment in SEGMENT_SPLIT.split(command):
        tokens = [t for t in segment.split() if t]
        if not tokens:
            continue
        paths = [p for p in (resolve_token(t) for t in tokens) if p is not None]
        if not any(touches_data(p) for p in paths):
            continue
        verb = Path(tokens[0].strip("\"'")).name.lower()
        if verb in LISTING_VERBS:
            # find -exec 처럼 다른 명령을 대신 실행하는 형태는 통과시키지 않는다.
            if UNSAFE_LISTING_FLAGS & set(tokens):
                return True
            continue
        return True
    return False


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            },
            ensure_ascii=False,
        )
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_name = str(payload.get("tool_name") or "")

    if tool_name == "Bash":
        command = str((payload.get("tool_input") or {}).get("command") or "")
        if not command or not blocked_bash(command):
            return 0
        deny(REASON_BASH)
        return 0

    path = target_path(payload)
    if path is None or not is_blocked(path):
        return 0
    deny(REASON_WRITE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
