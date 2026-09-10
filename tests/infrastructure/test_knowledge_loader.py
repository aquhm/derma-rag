"""지식데이터 `.doc` 로더 테스트.

실데이터를 쓰지 않는다. `tmp_path`에 같은 모양의 폴더 트리를 만들고 복합 파일을
조립해 넣는다. 조립 헬퍼는 test_doc_reader.py의 것을 그대로 쓴다.

핵심은 두 가지다. **같은 논문을 두 번 색인하지 않는가**(파일이 최상위와
`사용논문`에 중복으로 놓여 있다), 그리고 **깨진 파일 하나가 적재를 멈추지
않는가**.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.infrastructure.knowledge_loader import AihubKnowledgeLoader, paper_id
from tests.infrastructure.test_doc_reader import build_compound, build_word

# 실데이터와 같은 깊이로 만든다. 로더가 폴더 이름으로 거르기 때문이다.
KNOWLEDGE = "지식 데이터"


def write_doc(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    document, clx = build_word([(text, False)])
    path = folder / name
    path.write_bytes(build_compound({"WordDocument": document, "1Table": clx}))
    return path


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """최상위와 사용논문/미사용논문에 같은 파일이 놓인 트리."""
    root = tmp_path / KNOWLEDGE
    body = "각질층 수분 함량과 피부 장벽 기능의 관계를 " * 5

    write_doc(root / "국내석사", "가.doc", body + "가")
    write_doc(root / "해외저널", "나.doc", body + "나")
    write_doc(root / "사용논문" / "국내석사", "가.doc", body + "가")
    write_doc(root / "미사용논문" / "해외저널", "나.doc", body + "나")
    return tmp_path


def test_only_the_used_papers_folder_is_read(corpus: Path) -> None:
    # 최상위와 미사용논문에도 파일이 있지만 기본값은 사용논문만 본다.
    documents = list(AihubKnowledgeLoader(corpus).load())

    assert len(documents) == 1
    assert documents[0].id == paper_id("가.doc")


def test_reading_every_folder_deduplicates_by_file_name(corpus: Path) -> None:
    loader = AihubKnowledgeLoader(corpus, folder=None)

    documents = list(loader.load())

    # 파일은 4개지만 이름은 2종이다. 같은 논문을 두 번 색인하면 검색이 흐려진다.
    assert len(documents) == 2
    assert {document.id for document in documents} == {
        paper_id("가.doc"),
        paper_id("나.doc"),
    }
    assert len(loader.errors) == 2


def test_source_folder_is_kept_as_metadata(corpus: Path) -> None:
    documents = list(AihubKnowledgeLoader(corpus).load())

    assert documents[0].skin_detail == "국내석사"


def test_papers_have_no_skin_type(corpus: Path) -> None:
    # 논문에는 피부 유형 라벨이 없다. 없는 것을 지어내지 않는다.
    documents = list(AihubKnowledgeLoader(corpus).load())

    assert documents[0].skin_type is None


def test_body_text_is_carried(corpus: Path) -> None:
    documents = list(AihubKnowledgeLoader(corpus).load())

    assert "각질층 수분 함량" in documents[0].text


def test_limit_caps_the_number_of_documents(tmp_path: Path) -> None:
    folder = tmp_path / KNOWLEDGE / "사용논문" / "국내석사"
    body = "피부 장벽 연구 결과를 정리한 논문이다. " * 5
    for index in range(5):
        write_doc(folder, f"논문{index}.doc", body + str(index))

    documents = list(AihubKnowledgeLoader(tmp_path, limit=2).load())

    assert len(documents) == 2


def test_limit_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        AihubKnowledgeLoader(tmp_path, limit=0)


def test_missing_root_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="경로가 없다"):
        AihubKnowledgeLoader(tmp_path / "없는 폴더")


def test_no_files_is_an_error(tmp_path: Path) -> None:
    # 조용히 빈 목록을 내면 색인이 비어 있는 이유를 찾을 수 없다 (D-013과 같은 이유).
    (tmp_path / KNOWLEDGE).mkdir(parents=True)

    with pytest.raises(ValueError, match="파일이 없다"):
        list(AihubKnowledgeLoader(tmp_path).load())


# ---------------------------------------------------------------- 깨진 파일

def test_a_broken_file_does_not_stop_loading(tmp_path: Path) -> None:
    folder = tmp_path / KNOWLEDGE / "사용논문" / "국내석사"
    write_doc(folder, "정상.doc", "정상적인 논문 본문이 충분히 길게 들어 있다. " * 5)
    (folder / "깨짐.doc").write_bytes(b"{\\rtf1 not supported}")

    loader = AihubKnowledgeLoader(tmp_path)
    documents = list(loader.load())

    assert len(documents) == 1
    assert len(loader.errors) == 1


def test_error_messages_carry_no_body_text(tmp_path: Path) -> None:
    # 오류 기록에 원문이 새면 로그가 원본 유출 경로가 된다 (SAFETY.md S-6).
    folder = tmp_path / KNOWLEDGE / "사용논문" / "국내석사"
    folder.mkdir(parents=True, exist_ok=True)
    secret = "환자의 피부 상태 기록"
    document, _ = build_word([(secret, False)])
    # 테이블 스트림을 빼면 본문을 못 읽는다.
    (folder / "표없음.doc").write_bytes(build_compound({"WordDocument": document}))

    loader = AihubKnowledgeLoader(tmp_path)
    list(loader.load())

    assert loader.errors
    assert secret not in " ".join(loader.errors)


def test_too_short_text_is_skipped(tmp_path: Path) -> None:
    # 표지만 있는 파일이 섞여 있다. 색인에 넣어도 검색에 걸리지 않는다.
    folder = tmp_path / KNOWLEDGE / "사용논문" / "국내석사"
    write_doc(folder, "표지.doc", "제목")

    loader = AihubKnowledgeLoader(tmp_path)

    assert list(loader.load()) == []
    assert "표지.doc" in loader.errors[0]


# ------------------------------------------------------------------ 문서 ID

def test_same_file_name_gives_the_same_id() -> None:
    assert paper_id("피부 장벽.doc") == paper_id(" 피부 장벽.doc ")


def test_different_names_give_different_ids() -> None:
    assert paper_id("가.doc") != paper_id("나.doc")


def test_id_has_the_paper_prefix() -> None:
    # 발췌 ID(K-)와 구분된다. 색인에 무엇이 들었는지 ID만 보고 알 수 있어야 한다.
    assert paper_id("가.doc").startswith("P-")
