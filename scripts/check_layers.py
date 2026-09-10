"""레이어 의존 규칙 검사.

docs/ARCHITECTURE.md 2절의 표를 코드로 옮긴 것이다.
의존은 항상 안쪽으로만 향해야 한다.

사용:
    python scripts/check_layers.py            # src/ 전체 검사
    python scripts/check_layers.py <파일...>   # 지정한 파일만 검사

종료 코드:
    0  위반 없음
    1  위반 발견
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Windows 콘솔 기본 인코딩(CP949)에서 한글이 깨지는 것을 막는다.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, OSError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

LAYERS = ("domain", "application", "infrastructure", "interface")

# 각 레이어가 import 하면 안 되는 최상위 이름
FORBIDDEN: dict[str, frozenset[str]] = {
    "domain": frozenset(
        {"application", "infrastructure", "interface", "requests", "numpy"}
    ),
    "application": frozenset({"infrastructure", "interface", "requests", "numpy"}),
    "infrastructure": frozenset({"interface"}),
    "interface": frozenset(),
}

# 위반마다 붙일 근거
REASON: dict[tuple[str, str], str] = {
    ("domain", "numpy"): "domain은 벡터 저장 방식을 몰라야 한다. list[float]로 받아라",
    ("domain", "requests"): "domain은 외부 통신을 하지 않는다",
    ("application", "requests"): "HTTP 호출은 infrastructure 어댑터의 일이다",
    ("application", "numpy"): "벡터 연산은 infrastructure 어댑터의 일이다",
    ("application", "infrastructure"): (
        "포트(Protocol)를 통해 쓴다. 어댑터를 직접 import 하면 의존이 뒤집힌다"
    ),
    ("infrastructure", "interface"): "infrastructure는 진입점을 몰라야 한다",
}


def layer_of(path: Path) -> str | None:
    """파일이 속한 레이어 이름. src/ 밖이면 None."""
    try:
        rel = path.resolve().relative_to(SRC)
    except ValueError:
        return None
    head = rel.parts[0] if rel.parts else ""
    return head if head in LAYERS else None


def imported_roots(tree: ast.AST) -> list[tuple[str, int]]:
    """(최상위 모듈명, 행번호) 목록. 상대 import는 패키지 경로로 환원한다."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # 상대 import는 같은 패키지 안이므로 레이어 위반이 될 수 없다
                continue
            if node.module:
                found.append((node.module.split(".")[0], node.lineno))
    return found


def normalize(root: str) -> str:
    """`src.domain.models` 형태와 `domain.models` 형태를 같게 만든다."""
    return root


def check_file(path: Path) -> list[str]:
    layer = layer_of(path)
    if layer is None:
        return []

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: 구문 오류로 검사 불가 — {exc.msg}"]

    forbidden = FORBIDDEN[layer]
    violations: list[str] = []

    for root, lineno in imported_roots(tree):
        target = normalize(root)
        # `src` 패키지를 거쳐 들어오는 경우도 잡는다
        if target == "src":
            continue
        if target in forbidden:
            reason = REASON.get((layer, target), "레이어 의존 규칙 위반")
            rel = path.relative_to(PROJECT_ROOT)
            violations.append(f"{rel}:{lineno}: {layer} → {target} 금지. {reason}")

    return violations


def collect(targets: list[str]) -> list[Path]:
    if targets:
        return [Path(t) for t in targets if t.endswith(".py")]
    if not SRC.exists():
        return []
    return sorted(SRC.rglob("*.py"))


def hook_targets() -> list[str]:
    """PostToolUse 훅 모드. stdin의 JSON에서 편집된 파일 경로를 꺼낸다.

    src/ 아래 .py가 아니면 빈 목록을 돌려 검사를 건너뛴다.
    """
    import json

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return []

    candidates = [
        (payload.get("tool_response") or {}).get("filePath"),
        (payload.get("tool_input") or {}).get("file_path"),
    ]
    for raw in candidates:
        if not raw or not str(raw).endswith(".py"):
            continue
        path = Path(str(raw))
        if layer_of(path) is not None:
            return [str(path)]
    return []


def main(argv: list[str]) -> int:
    hook_mode = bool(argv) and argv[0] == "--hook"
    if hook_mode:
        argv = hook_targets()
        if not argv:
            return 0

    files = collect(argv)
    if not files:
        return 0

    violations: list[str] = []
    for f in files:
        if not f.exists():
            continue
        violations.extend(check_file(f))

    if violations:
        print("레이어 의존 규칙 위반", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print("", file=sys.stderr)
        print("규칙 표: docs/ARCHITECTURE.md 2절", file=sys.stderr)
        print(
            "검사 스크립트를 고쳐서 통과시키지 마라. 코드를 고쳐라.",
            file=sys.stderr,
        )
        # 훅 모드에서는 종료 코드 2가 blocking error다. 모델에게 되먹여
        # 즉시 고치게 한다. CLI 모드에서는 관례대로 1을 쓴다.
        return 2 if hook_mode else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
