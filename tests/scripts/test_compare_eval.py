"""compare_eval.py 테스트.

케이스 파일은 CLI가 만든 형식을 그대로 쓴다. 원문은 들어 있지 않다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_eval import compare, load_cases


def row(
    case_id: str,
    outcome: str = "answered",
    accuracy: float | None = 0.5,
    detail: str = "",
) -> dict:
    return {
        "case_id": case_id,
        "outcome": outcome,
        "accuracy": accuracy,
        "similarity": None,
        "detail": detail,
    }


def write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def as_map(rows: list[dict]) -> dict:
    return {r["case_id"]: r for r in rows}


# ------------------------------------------------------------------- 로드

def test_cases_are_keyed_by_id(tmp_path: Path) -> None:
    path = write(tmp_path / "a.json", [row("Q-1"), row("Q-2")])

    assert sorted(load_cases(path)) == ["Q-1", "Q-2"]


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="케이스 파일이 없다"):
        load_cases(tmp_path / "없음.json")


def test_wrong_shape_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"case_id": "Q-1"}), encoding="utf-8")

    with pytest.raises(ValueError, match="목록이 아니다"):
        load_cases(path)


# ------------------------------------------------------------------- 비교

def test_only_common_cases_are_compared() -> None:
    before = as_map([row("Q-1", accuracy=0.4), row("Q-2", accuracy=0.9)])
    after = as_map([row("Q-1", accuracy=0.6), row("Q-3", accuracy=0.1)])

    report = compare(before, after)

    assert "공통 1건" in report
    assert "+0.2000" in report


def test_cases_answered_on_only_one_side_are_excluded_from_f1() -> None:
    # 한쪽만 답한 건을 섞으면 "거부가 줄어 평균이 내려간" 것과 "답이 나빠진" 것이
    # 구분되지 않는다.
    before = as_map([row("Q-1", accuracy=0.4), row("Q-2", "refused", None, "근거 없음")])
    after = as_map([row("Q-1", accuracy=0.4), row("Q-2", accuracy=0.1)])

    report = compare(before, after)

    assert "(1건)" in report


def test_outcome_transitions_are_reported() -> None:
    before = as_map([row("Q-1", "refused", None, "근거 없음"), row("Q-2", "failed", None, "Timeout")])
    after = as_map([row("Q-1", accuracy=0.3), row("Q-2", accuracy=0.2)])

    report = compare(before, after)

    assert "refused → answered: 1건" in report
    assert "failed → answered: 1건" in report


def test_unchanged_outcomes_say_so() -> None:
    before = as_map([row("Q-1")])
    after = as_map([row("Q-1")])

    assert "결과가 바뀐 케이스: 없음" in compare(before, after)


def test_no_common_cases_is_reported() -> None:
    report = compare(as_map([row("Q-1")]), as_map([row("Q-2")]))

    assert "공통 케이스가 없다" in report


def test_no_scored_cases_is_reported() -> None:
    before = as_map([row("Q-1", "refused", None, "근거 없음")])
    after = as_map([row("Q-1", "refused", None, "근거 없음")])

    assert "양쪽 다 채점된 건이 없다" in compare(before, after)
