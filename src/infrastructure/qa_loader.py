"""평가셋 로더.

평가 입력 형식은 이 프로젝트가 정한다. AI Hub QA쌍 10,043쌍을 이 형식으로
바꾸는 변환기는 원본 필드명을 확인한 뒤(M2-1) M2-5에서 만든다. 형식을 먼저
고정해 두면 그때 할 일이 "변환" 하나로 줄어든다 (MEMORY.md D-025).

    [
      {
        "id": "Q-1",
        "question": "질문 문장",
        "expected": "기대 답변 (선택)",
        "gold_doc_ids": ["sample-001"]
      }
    ]

expected와 gold_doc_ids는 없어도 된다. gold_doc_ids가 없으면 Recall 채점에서
제외된다 (EvalCase.is_gradable).

예외 메시지에 질문이나 기대 답변을 넣지 않는다. 평가셋에는 실데이터에서 나온
문장이 들어간다 (SAFETY.md S-6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.domain.models import EvalCase


class JsonQaSetLoader:
    """평가셋 JSON 하나를 EvalCase 목록으로 읽는다."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[EvalCase]:
        if not self._path.is_file():
            raise FileNotFoundError(f"평가셋 파일이 없다: {self._path}")

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"평가셋이 올바른 JSON이 아니다: {self._path} ({exc.lineno}번째 줄)"
            ) from exc

        if not isinstance(raw, list):
            raise ValueError(f"평가셋 최상위가 목록이 아니다: {self._path}")

        cases = [self._to_case(record, position) for position, record in enumerate(raw, 1)]

        if not cases:
            raise ValueError(f"평가 케이스가 없다: {self._path}")

        self._reject_duplicates(cases)
        return cases

    def _to_case(self, record: Any, position: int) -> EvalCase:
        where = f"{position}번째 케이스"

        if not isinstance(record, dict):
            raise ValueError(f"{where}가 객체가 아니다")

        for key in ("id", "question"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                raise ValueError(f"{where}에 {key}가 없거나 비어 있다")

        gold = record.get("gold_doc_ids", [])
        if not isinstance(gold, list) or any(not isinstance(g, str) for g in gold):
            raise ValueError(f"{where}의 gold_doc_ids가 문자열 목록이 아니다")

        expected = record.get("expected", "")
        if not isinstance(expected, str):
            raise ValueError(f"{where}의 expected가 문자열이 아니다")

        return EvalCase(
            id=record["id"],
            question=record["question"],
            expected=expected,
            gold_doc_ids=tuple(gold),
        )

    @staticmethod
    def _reject_duplicates(cases: list[EvalCase]) -> None:
        seen: set[str] = set()
        duplicated: set[str] = set()
        for case in cases:
            if case.id in seen:
                duplicated.add(case.id)
            seen.add(case.id)
        if duplicated:
            raise ValueError(f"평가 케이스 ID가 중복이다: {sorted(duplicated)}")
