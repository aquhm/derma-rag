"""실험 두 개를 공통 케이스 기준으로 비교한다 (미해결 O-5).

요약 숫자만 보면 채점 대상이 달라졌을 때 두 실험을 견줄 수 없다. 실제로 M3-4a에서
기준선은 182건, 실험은 200건을 채점해 F1 비교가 흐려졌다.

입력은 `python -m src.cli eval --out`이 함께 남기는 케이스 파일이다.

    .venv/Scripts/python.exe scripts/compare_eval.py eval/raw/baseline.cases.json eval/raw/실험.cases.json

케이스 파일에는 케이스 ID와 점수만 있다. 질문과 답변 원문은 들어 있지 않다
(SAFETY.md S-7).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

OUTCOMES = ("answered", "refused", "failed")


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"케이스 파일이 없다: {path}")

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"케이스 파일이 목록이 아니다: {path}")

    return {row["case_id"]: row for row in rows}


def compare(before: dict[str, Any], after: dict[str, Any]) -> str:
    """공통 케이스만 골라 지표를 견준다."""
    common = sorted(set(before) & set(after))
    lines = [
        f"전체: 이전 {len(before)}건, 이후 {len(after)}건, 공통 {len(common)}건",
        "",
    ]
    if not common:
        lines.append("공통 케이스가 없다. 표본이나 seed가 다르다.")
        return "\n".join(lines)

    lines.append(f"{'지표':<22}{'이전':>10}{'이후':>10}{'차이':>10}")
    lines.append(_metric_line("F1 (공통, 양쪽 답변)", before, after, common))
    lines.append(_outcome_line(before, after, common))
    lines.append("")
    lines += _transitions(before, after, common)
    return "\n".join(lines)


def _metric_line(
    label: str, before: dict[str, Any], after: dict[str, Any], common: list[str]
) -> str:
    """양쪽 모두 답한 케이스의 F1 평균만 비교한다.

    한쪽만 답한 케이스를 섞으면 "거부가 줄어서 평균이 내려간" 것인지 "답이
    나빠진" 것인지 구분되지 않는다.
    """
    pairs = [
        (before[cid]["accuracy"], after[cid]["accuracy"])
        for cid in common
        if before[cid].get("accuracy") is not None
        and after[cid].get("accuracy") is not None
    ]
    if not pairs:
        return f"{label:<22}{'—':>10}{'—':>10}{'—':>10}  (양쪽 다 채점된 건이 없다)"

    left = statistics.mean(p[0] for p in pairs)
    right = statistics.mean(p[1] for p in pairs)
    return (
        f"{label:<22}{left:>10.4f}{right:>10.4f}{right - left:>+10.4f}"
        f"  ({len(pairs)}건)"
    )


def _outcome_line(
    before: dict[str, Any], after: dict[str, Any], common: list[str]
) -> str:
    left = Counter(before[cid]["outcome"] for cid in common)
    right = Counter(after[cid]["outcome"] for cid in common)
    parts = [
        f"{name} {left[name]}→{right[name]}"
        for name in OUTCOMES
        if left[name] or right[name]
    ]
    return f"{'결과 분포':<22}{'  '.join(parts)}"


def _transitions(
    before: dict[str, Any], after: dict[str, Any], common: list[str]
) -> list[str]:
    """무엇이 무엇으로 바뀌었는지. 어느 방향으로 움직였는지가 판정의 핵심이다."""
    moves = Counter(
        (before[cid]["outcome"], after[cid]["outcome"])
        for cid in common
        if before[cid]["outcome"] != after[cid]["outcome"]
    )
    if not moves:
        return ["결과가 바뀐 케이스: 없음"]

    lines = ["결과가 바뀐 케이스:"]
    lines += [
        f"  {source} → {target}: {count}건"
        for (source, target), count in sorted(moves.items(), key=lambda m: -m[1])
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="실험 두 개를 공통 케이스 기준으로 비교한다 (원문은 다루지 않는다)"
    )
    parser.add_argument("before", type=Path, help="이전 실험의 케이스 파일")
    parser.add_argument("after", type=Path, help="이후 실험의 케이스 파일")
    args = parser.parse_args(argv)

    print(f"이전: {args.before}")
    print(f"이후: {args.after}")
    print()
    print(compare(load_cases(args.before), load_cases(args.after)))
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    raise SystemExit(main())
