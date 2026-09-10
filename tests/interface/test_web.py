"""로컬 웹 UI 테스트.

HTTP 배선은 표준 라이브러리에 맡기고, 여기서는 순수 함수만 검증한다.
스모크 수준으로 둔다 (docs/guidelines/04-interface.md).

가장 중요한 것은 마지막 절이다. **판정 기록에 질문과 답변이 남지 않는지**
확인한다 (SAFETY.md S-7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.models import Answer, Chunk, Evidence, Refusal, RefusalReason, SkinType
from src.domain.policy import DISCLAIMER
from src.interface.web import (
    PAGE,
    answer_payload,
    append_judgment,
    judgment_row,
)


def evidence(score: float = 0.72) -> Evidence:
    return Evidence(
        chunk=Chunk(
            id="A000175_07_QA2#0",
            doc_id="A000175_07_QA2",
            text="질문: 볼이 붉어집니다\n답변: 열감을 가라앉히라고 기술된다",
            skin_type=SkinType.SENSITIVE,
            area="볼",
            skin_detail="홍조 피부",
        ),
        score=score,
    )


# ------------------------------------------------------------- 답변 페이로드

def test_answer_payload_carries_body_evidence_and_disclaimer() -> None:
    payload = answer_payload(Answer(body="본문입니다", evidence=(evidence(),)))

    assert payload["kind"] == "answer"
    assert payload["body"] == "본문입니다"
    assert payload["disclaimer"] == DISCLAIMER
    assert payload["evidence"][0]["id"] == "A000175_07_QA2#0"


def test_answer_payload_includes_scores_and_labels() -> None:
    payload = answer_payload(Answer(body="본문", evidence=(evidence(0.72),)))

    row = payload["evidence"][0]
    assert row["score"] == pytest.approx(0.72)
    assert row["skin_type"] == "민감성"
    assert row["skin_detail"] == "홍조 피부"
    assert row["area"] == "볼"


def test_refusal_payload_shows_reason_and_detail() -> None:
    payload = answer_payload(
        Refusal(reason=RefusalReason.NO_EVIDENCE, detail="검색 결과가 없다")
    )

    assert payload["kind"] == "refusal"
    assert payload["reason"] == "근거 없음"
    assert payload["detail"] == "검색 결과가 없다"
    assert payload.get("evidence", []) == []


def test_refusal_payload_has_no_disclaimer() -> None:
    # 면책 문구는 Answer에만 붙는다. 거부에 붙이면 답을 한 것처럼 보인다.
    payload = answer_payload(Refusal(reason=RefusalReason.SAFETY_VIOLATION))

    assert "disclaimer" not in payload


# ------------------------------------------------------------------ 판정 기록

def test_judgment_row_keeps_verdict_and_chunk_ids() -> None:
    row = judgment_row("O", answer_payload(Answer(body="본문", evidence=(evidence(),))))

    assert row["verdict"] == "O"
    assert row["chunk_ids"] == ["A000175_07_QA2#0"]
    assert row["scores"] == [pytest.approx(0.72)]
    assert row["body_length"] == 2


def test_judgment_row_records_refusals() -> None:
    row = judgment_row(
        "O", answer_payload(Refusal(reason=RefusalReason.NO_EVIDENCE, detail="없다"))
    )

    assert row["kind"] == "refusal"
    assert row["reason"] == "근거 없음"


@pytest.mark.parametrize("verdict", ["O", "△", "X"])
def test_all_three_verdicts_are_allowed(verdict: str) -> None:
    row = judgment_row(verdict, answer_payload(Answer(body="본문", evidence=(evidence(),))))

    assert row["verdict"] == verdict


def test_unknown_verdict_is_rejected() -> None:
    with pytest.raises(ValueError, match="판정"):
        judgment_row("좋음", answer_payload(Answer(body="본문", evidence=(evidence(),))))


def test_judgment_is_appended_one_line_per_row(tmp_path: Path) -> None:
    path = tmp_path / "raw" / "human-check.jsonl"

    append_judgment(path, {"verdict": "O"})
    append_judgment(path, {"verdict": "X"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["verdict"] for line in lines] == ["O", "X"]


def test_judgment_row_has_a_timestamp() -> None:
    row = judgment_row("O", answer_payload(Answer(body="본문", evidence=(evidence(),))))

    assert row["timestamp"]


# ----------------------------------------- 질문·답변 원문 미저장 (S-7, S-6)

def test_judgment_row_never_contains_the_question_or_answer() -> None:
    # 질문은 개인 건강 정보일 수 있고, 답변에는 AI Hub 원문이 섞인다.
    body = "볼의 열감을 먼저 가라앉히라고 기술돼 있습니다"
    payload = answer_payload(Answer(body=body, evidence=(evidence(),)))

    row = str(judgment_row("O", payload))

    assert body not in row
    assert "질문: 볼이 붉어집니다" not in row


def test_page_does_not_embed_any_dataset_text() -> None:
    # 페이지는 정적이다. 데이터는 실행 중에 받아 화면에만 그린다.
    assert "질문: 볼이" not in PAGE
    assert "답변: 열감" not in PAGE


def test_page_warns_that_answers_are_not_medical_advice() -> None:
    assert "진단" in PAGE


# ------------------------------------------- 질문 유형 태그 (A안, D-034)

def test_judgment_row_keeps_the_question_kind() -> None:
    # 유형은 분류일 뿐 질문 원문이 아니다. S-7에 걸리지 않는다.
    row = judgment_row(
        "O",
        answer_payload(Answer(body="본문", evidence=(evidence(),))),
        question_kind="진단요구",
    )

    assert row["question_kind"] == "진단요구"


def test_question_kind_defaults_to_unknown() -> None:
    row = judgment_row("O", answer_payload(Answer(body="본문", evidence=(evidence(),))))

    assert row["question_kind"] == "미분류"


def test_unknown_question_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="유형"):
        judgment_row(
            "O",
            answer_payload(Answer(body="본문", evidence=(evidence(),))),
            question_kind="아무거나",
        )


def test_page_offers_the_question_kinds() -> None:
    for kind in ("정상", "무관", "진단요구"):
        assert kind in PAGE
