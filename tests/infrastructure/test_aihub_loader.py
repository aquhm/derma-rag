"""AI Hub 실데이터 로더 테스트 (M2-2, M2-3, M2-5).

실데이터를 쓰지 않는다. 실제 구조만 흉내 낸 가짜 JSON을 tmp_path에 만든다
(MEMORY.md D-006).

색인 대상은 QA 라벨링 JSON 안의 근거 발췌다 (D-027). 같은 파서가 평가셋
변환에도 쓰인다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.models import SkinType
from src.infrastructure.aihub_loader import (
    AihubQaLoader,
    AihubQaPairLoader,
    AihubQaSetLoader,
    knowledge_id,
)


def qa_record(
    seq: str = "A000137_07_QA1",
    question: str = "민감성 피부인데 어떻게 하나요?",
    answer: str = "자극을 줄이라고 기술된다.",
    knowledge: list[str] | None = None,
    area: str = "얼굴 전체",
) -> dict:
    record: dict = {
        "Data_info": {"SEQ": seq, "License": "㈜데이터쿡"},
        "Annotation_info": {"User Question": question, "Makeup Response": answer},
        "Human_info": {"Makeup focus areas": area, "Age": "30대"},
        "Skin_info": {"Skin condition category": "지성"},
    }
    texts = knowledge if knowledge is not None else ["근거 문단 하나"]
    for number in range(1, 6):
        text = texts[number - 1] if number <= len(texts) else ""
        record[f"Source_info{number}"] = {
            f"Knowledge Data{number}": text,
            f"File Name{number}_1": f"원본{number}.doc" if text else "",
        }
    return record


def write_qa(root: Path, folder: str, name: str, record: dict) -> Path:
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


# ------------------------------------------------------------- 지식 ID

def test_same_text_gives_the_same_id() -> None:
    assert knowledge_id("같은 문단") == knowledge_id("같은 문단")


def test_different_text_gives_a_different_id() -> None:
    assert knowledge_id("문단 A") != knowledge_id("문단 B")


def test_id_ignores_surrounding_whitespace() -> None:
    # 같은 근거가 파일마다 공백만 다르게 들어 있다. 다른 문서로 세면 안 된다.
    assert knowledge_id("  문단  ") == knowledge_id("문단")


def test_id_is_short_and_prefixed() -> None:
    generated = knowledge_id("문단")

    assert generated.startswith("K-")
    assert len(generated) <= 16


# --------------------------------------------------- AihubQaLoader (M2-2)

def test_every_knowledge_excerpt_becomes_a_document(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_아토피 피부", "1.json",
             qa_record(knowledge=["근거 하나", "근거 둘"]))

    documents = list(AihubQaLoader(tmp_path).load())

    assert {d.text for d in documents} == {"근거 하나", "근거 둘"}


def test_duplicate_excerpts_are_loaded_once(tmp_path: Path) -> None:
    # 고유 5,921건 대 전체 9,181건. 중복을 그대로 색인하면 검색 결과가 같은 문단으로
    # 채워지고 임베딩 비용도 55% 늘어난다.
    write_qa(tmp_path, "TL_민감성 피부_아토피 피부", "1.json", qa_record(knowledge=["같은 근거"]))
    write_qa(tmp_path, "TL_민감성 피부_아토피 피부", "2.json",
             qa_record(seq="A2_07_QA1", knowledge=["같은 근거"]))

    documents = list(AihubQaLoader(tmp_path).load())

    assert len(documents) == 1


def test_empty_knowledge_slots_are_skipped(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_아토피 피부", "1.json", qa_record(knowledge=["근거"]))

    documents = list(AihubQaLoader(tmp_path).load())

    assert len(documents) == 1


def test_folder_name_becomes_skin_type_and_detail(tmp_path: Path) -> None:
    # D-028. 폴더명이 4대 분류와 세부 유형을 함께 담는다.
    write_qa(tmp_path, "TL_염증성 피부_지루성 피부염", "1.json", qa_record())

    document = next(iter(AihubQaLoader(tmp_path).load()))

    assert document.skin_type is SkinType.INFLAMMATORY
    assert document.skin_detail == "지루성 피부염"


@pytest.mark.parametrize(
    "folder,expected",
    [
        ("TL_민감성 피부_홍조 피부", SkinType.SENSITIVE),
        ("TL_염증성 피부_주사 피부", SkinType.INFLAMMATORY),
        ("VL_색소문제 피부_기미,주근깨 피부", SkinType.PIGMENT),
        ("VL_조직변화 피부_노화 피부", SkinType.TEXTURE),
    ],
)
def test_all_four_categories_are_mapped(
    tmp_path: Path, folder: str, expected: SkinType
) -> None:
    write_qa(tmp_path, folder, "1.json", qa_record())

    assert next(iter(AihubQaLoader(tmp_path).load())).skin_type is expected


def test_unknown_folder_name_leaves_skin_type_empty(tmp_path: Path) -> None:
    # 분류에 없는 표기가 나와도 적재가 통째로 실패하면 안 된다 (D-028).
    write_qa(tmp_path, "TL_새로운 분류_새 유형", "1.json", qa_record())

    document = next(iter(AihubQaLoader(tmp_path).load()))

    assert document.skin_type is None
    assert document.skin_detail == "새 유형"


def test_area_comes_from_human_info(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(area="볼"))

    assert next(iter(AihubQaLoader(tmp_path).load())).area == "볼"


def test_non_qa_folders_are_ignored(tmp_path: Path) -> None:
    # 지식데이터 메타 JSON 3,090개는 근거를 담지 않는다. 읽을 이유가 없다.
    (tmp_path / "json").mkdir()
    (tmp_path / "json" / "meta.json").write_text(
        json.dumps({"Data_info": {"Title": "제목"}}), encoding="utf-8"
    )
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record())

    assert len(list(AihubQaLoader(tmp_path).load())) == 1


def test_broken_file_is_skipped_and_counted(tmp_path: Path) -> None:
    # 9,031개 중 하나가 깨졌다고 적재 전체가 멈추면 안 된다.
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record())
    (tmp_path / "TL_민감성 피부_홍조 피부" / "2.json").write_text("{깨진", encoding="utf-8")

    loader = AihubQaLoader(tmp_path)
    documents = list(loader.load())

    assert len(documents) == 1
    assert len(loader.errors) == 1
    assert "2.json" in loader.errors[0]


def test_missing_root_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="경로가 없다"):
        AihubQaLoader(tmp_path / "없음")


def test_no_qa_files_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "빈폴더").mkdir()

    with pytest.raises(ValueError, match="QA 파일이 없다"):
        list(AihubQaLoader(tmp_path).load())


# ------------------------------------------------ AihubQaSetLoader (M2-5)

def test_each_file_becomes_one_eval_case(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(seq="A1_07_QA1"))
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "2.json", qa_record(seq="A2_07_QA1"))

    cases = AihubQaSetLoader(tmp_path).load()

    assert [c.id for c in cases] == ["A1_07_QA1", "A2_07_QA1"]


def test_question_and_expected_answer_are_carried(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json",
             qa_record(question="세안 후 당깁니다", answer="보습을 권한다고 기술된다"))

    case = AihubQaSetLoader(tmp_path).load()[0]

    assert case.question == "세안 후 당깁니다"
    assert case.expected == "보습을 권한다고 기술된다"


def test_gold_ids_match_the_loaded_documents(tmp_path: Path) -> None:
    # 이것이 Recall@k 자동 채점의 근거다. 두 로더가 같은 ID 규칙을 써야 한다.
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json",
             qa_record(knowledge=["근거 하나", "근거 둘"]))

    documents = list(AihubQaLoader(tmp_path).load())
    case = AihubQaSetLoader(tmp_path).load()[0]

    assert set(case.gold_doc_ids) == {d.id for d in documents}
    assert len(case.gold_doc_ids) == 2


def test_case_id_falls_back_to_the_file_name(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "없는SEQ.json", qa_record(seq=""))

    assert AihubQaSetLoader(tmp_path).load()[0].id == "없는SEQ"


def test_duplicate_seq_values_stay_unique(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(seq="같은SEQ"))
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "2.json", qa_record(seq="같은SEQ"))

    cases = AihubQaSetLoader(tmp_path).load()

    assert len({c.id for c in cases}) == 2


def test_case_without_a_question_is_skipped(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(question="   "))
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "2.json", qa_record(seq="A2_07_QA1"))

    cases = AihubQaSetLoader(tmp_path).load()

    assert [c.id for c in cases] == ["A2_07_QA1"]


def test_error_messages_do_not_carry_the_question(tmp_path: Path) -> None:
    question = "민감성 피부가 세안 후 당길 때 무엇을 주의해야 하나요?"
    path = write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(question=question))
    path.write_text(path.read_text(encoding="utf-8")[:-5], encoding="utf-8")

    loader = AihubQaSetLoader(tmp_path)
    loader.load()

    assert all(question not in message for message in loader.errors)


# ------------------------------------------ AihubQaPairLoader (M3-6, D-032)

def test_each_qa_becomes_one_document(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(seq="A1_07_QA1"))
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "2.json", qa_record(seq="A2_07_QA1"))

    documents = list(AihubQaPairLoader(tmp_path).load())

    assert [d.id for d in documents] == ["A1_07_QA1", "A2_07_QA1"]


def test_document_text_holds_question_and_answer(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json",
             qa_record(question="세안 후 당깁니다", answer="보습을 권한다고 기술된다"))

    text = next(iter(AihubQaPairLoader(tmp_path).load())).text

    assert "세안 후 당깁니다" in text
    assert "보습을 권한다고 기술된다" in text


def test_pair_document_carries_the_folder_labels(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_염증성 피부_지루성 피부염", "1.json", qa_record(area="턱"))

    document = next(iter(AihubQaPairLoader(tmp_path).load()))

    assert document.skin_type is SkinType.INFLAMMATORY
    assert document.skin_detail == "지루성 피부염"
    assert document.area == "턱"


def test_qa_without_an_answer_is_skipped(tmp_path: Path) -> None:
    # 답변이 없으면 색인에 넣을 내용이 질문뿐이다. 근거로 쓸 수 없다.
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(answer="   "))
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "2.json", qa_record(seq="A2_07_QA1"))

    assert [d.id for d in AihubQaPairLoader(tmp_path).load()] == ["A2_07_QA1"]


def test_pair_ids_match_the_eval_case_ids(tmp_path: Path) -> None:
    # 평가에서 색인에 든 케이스를 빼려면 두 로더의 ID 규칙이 같아야 한다.
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record(seq="같은SEQ"))
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "2.json", qa_record(seq="같은SEQ"))

    documents = [d.id for d in AihubQaPairLoader(tmp_path).load()]
    cases = [c.id for c in AihubQaSetLoader(tmp_path).load()]

    assert documents == cases


def test_pair_loader_reports_broken_files(tmp_path: Path) -> None:
    write_qa(tmp_path, "TL_민감성 피부_홍조 피부", "1.json", qa_record())
    (tmp_path / "TL_민감성 피부_홍조 피부" / "2.json").write_text("{깨진", encoding="utf-8")

    loader = AihubQaPairLoader(tmp_path)
    documents = list(loader.load())

    assert len(documents) == 1
    assert len(loader.errors) == 1


def test_pair_loader_needs_qa_files(tmp_path: Path) -> None:
    (tmp_path / "빈폴더").mkdir()

    with pytest.raises(ValueError, match="QA 파일이 없다"):
        list(AihubQaPairLoader(tmp_path).load())
