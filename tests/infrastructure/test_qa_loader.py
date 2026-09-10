"""JsonQaSetLoader 테스트.

평가 입력 형식(eval/qaset.json)을 읽어 EvalCase로 바꾼다. AI Hub QA쌍을 이
형식으로 변환하는 일은 M2-5에서 원본 필드명을 확인한 뒤에 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infrastructure.qa_loader import JsonQaSetLoader


def write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def case(case_id: str = "Q-1", **overrides: object) -> dict:
    record = {
        "id": case_id,
        "question": "민감성 피부는 무엇을 주의해야 하나요?",
        "expected": "자극 요소를 줄이라고 기술돼 있다.",
        "gold_doc_ids": ["sample-001"],
    }
    record.update(overrides)
    return record


def test_reads_every_case(tmp_path: Path) -> None:
    path = write(tmp_path / "qaset.json", [case("Q-1"), case("Q-2")])

    cases = JsonQaSetLoader(path).load()

    assert [c.id for c in cases] == ["Q-1", "Q-2"]


def test_gold_doc_ids_become_a_tuple(tmp_path: Path) -> None:
    path = write(tmp_path / "qaset.json", [case(gold_doc_ids=["a", "b"])])

    assert JsonQaSetLoader(path).load()[0].gold_doc_ids == ("a", "b")


def test_expected_and_gold_are_optional(tmp_path: Path) -> None:
    path = write(tmp_path / "qaset.json", [{"id": "Q-1", "question": "질문"}])

    loaded = JsonQaSetLoader(path).load()[0]

    assert loaded.expected == ""
    assert loaded.gold_doc_ids == ()
    assert loaded.is_gradable is False


def test_missing_file_is_reported_with_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="평가셋 파일"):
        JsonQaSetLoader(tmp_path / "없음.json").load()


def test_top_level_must_be_a_list(tmp_path: Path) -> None:
    path = write(tmp_path / "qaset.json", {"cases": [case()]})

    with pytest.raises(ValueError, match="목록"):
        JsonQaSetLoader(path).load()


def test_broken_json_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "qaset.json"
    path.write_text("{깨진", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        JsonQaSetLoader(path).load()


def test_missing_question_reports_the_position(tmp_path: Path) -> None:
    path = write(tmp_path / "qaset.json", [case("Q-1"), {"id": "Q-2"}])

    with pytest.raises(ValueError, match="2번째"):
        JsonQaSetLoader(path).load()


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    # 같은 ID가 둘이면 결과를 케이스별로 되짚을 수 없다.
    path = write(tmp_path / "qaset.json", [case("Q-1"), case("Q-1")])

    with pytest.raises(ValueError, match="중복"):
        JsonQaSetLoader(path).load()


def test_gold_doc_ids_must_be_a_list_of_strings(tmp_path: Path) -> None:
    path = write(tmp_path / "qaset.json", [case(gold_doc_ids="sample-001")])

    with pytest.raises(ValueError, match="gold_doc_ids"):
        JsonQaSetLoader(path).load()


def test_empty_set_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path / "qaset.json", [])

    with pytest.raises(ValueError, match="평가 케이스가 없다"):
        JsonQaSetLoader(path).load()


def test_error_message_does_not_carry_the_question_text(tmp_path: Path) -> None:
    # 평가셋에는 실데이터에서 나온 질문이 들어간다. 예외 메시지로 새면 로그에
    # 남는다 (SAFETY.md S-6).
    question = "민감성 피부가 세안 후 당길 때 무엇을 주의해야 하나요?"
    path = write(
        tmp_path / "qaset.json", [{"id": "Q-1", "question": question, "gold_doc_ids": 3}]
    )

    with pytest.raises(ValueError) as caught:
        JsonQaSetLoader(path).load()

    assert question not in str(caught.value)


def test_sample_qaset_in_the_repository_loads() -> None:
    # eval/qaset.sample.json은 이 프로젝트가 지어낸 문장이다 (D-014와 같은 취급).
    path = Path(__file__).resolve().parents[2] / "eval" / "qaset.sample.json"

    cases = JsonQaSetLoader(path).load()

    assert len(cases) >= 5
    assert all(c.is_gradable for c in cases)
