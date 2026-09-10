"""AI Hub 실데이터 로더 (M2-2, M2-3, M2-5).

색인 대상은 QA 라벨링 JSON 안의 근거 발췌다(`Source_info1~5.Knowledge Data1~5`).
지식데이터 `.doc` 6,170건은 형식이 섞여 있어 지금 쓰지 않는다. 근거는
`MEMORY.md` D-027.

두 로더가 같은 파일을 읽는다.

- `AihubQaLoader` — 근거 발췌를 `Document`로 만든다. 색인 입력 (D-027).
- `AihubQaPairLoader` — QA쌍을 `Document` 하나로 만든다. 색인 입력 (D-032).
- `AihubQaSetLoader` — 질문과 기대 답변을 `EvalCase`로 만든다. 평가 입력이다.

**두 로더는 같은 ID 규칙(`knowledge_id`)을 쓴다.** 이것이 Recall@k 자동 채점의
근거다. 규칙이 갈리면 정답 문서를 찾아도 맞혔다고 세지 못한다.

메타데이터는 폴더명에서 나온다. `TL_민감성 피부_아토피 피부` 형태로 4대 분류와
세부 유형이 함께 들어 있다 (D-028).

파일 하나가 깨져도 적재를 멈추지 않는다. 9,031개 중 하나 때문에 전체가 실패하면
곤란하다. 건너뛴 파일은 `errors`에 파일명만 남긴다. 원문은 남기지 않는다
(SAFETY.md S-6).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.domain.models import Document, EvalCase, SkinType

# QA 라벨링 폴더 접두사. Training과 Validation이다.
QA_FOLDER_PREFIXES = ("TL_", "VL_")

# 한 QA에 달릴 수 있는 근거 칸 수. 실측 최대 4개가 채워져 있었다.
MAX_SOURCES = 5

# 폴더명의 4대 분류 표기 → SkinType. 표기가 SkinType 값과 조금 다르다
# ("민감성 피부" 대 "민감성").
CATEGORY_TO_SKIN_TYPE = {
    "민감성 피부": SkinType.SENSITIVE,
    "염증성 피부": SkinType.INFLAMMATORY,
    "색소문제 피부": SkinType.PIGMENT,
    "조직변화 피부": SkinType.TEXTURE,
}

# 해시에서 쓸 자릿수. 5,921건 규모에서 12자리 충돌 확률은 무시할 수 있다.
ID_DIGITS = 12


def knowledge_id(text: str) -> str:
    """근거 발췌의 문서 ID. 같은 내용이면 같은 ID가 나온다.

    내용 해시를 쓰는 이유는 중복 제거와 정답 연결을 한 번에 풀기 위해서다. 같은
    근거가 여러 QA에 실려 있고(9,181건 중 고유 5,921건), 파일명 대조로 맞추면
    표기 차이에서 어긋난다.
    """
    digest = hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
    return f"K-{digest[:ID_DIGITS]}"


class _QaReader:
    """QA 라벨링 JSON을 훑는 공통 부분."""

    def __init__(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"경로가 없다: {root}")
        self._root = root
        self.errors: list[str] = []

    def _files(self) -> list[Path]:
        return sorted(
            path
            for path in self._root.rglob("*.json")
            if path.parent.name.startswith(QA_FOLDER_PREFIXES)
        )

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            # 메시지에 파일 이름과 예외 종류만 넣는다. 내용은 넣지 않는다.
            self.errors.append(f"{path.name}: {type(exc).__name__}")
            return None
        if not isinstance(payload, dict):
            self.errors.append(f"{path.name}: 최상위가 객체가 아니다")
            return None
        return payload

    @staticmethod
    def _knowledge_texts(record: dict[str, Any]) -> list[str]:
        texts = []
        for number in range(1, MAX_SOURCES + 1):
            source = record.get(f"Source_info{number}")
            if not isinstance(source, dict):
                continue
            text = source.get(f"Knowledge Data{number}", "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        return texts


class AihubQaLoader(_QaReader):
    """application.ports.DocumentLoader를 만족한다.

    근거 발췌 하나가 Document 하나다. 같은 발췌가 여러 QA에 실려 있으면 한 번만
    내보낸다.
    """

    def load(self) -> Iterator[Document]:
        files = self._files()
        if not files:
            raise ValueError(f"QA 파일이 없다: {self._root}")

        seen: set[str] = set()
        for path in files:
            record = self._read(path)
            if record is None:
                continue

            skin_type, skin_detail = _labels_from_folder(path.parent.name)
            area = _text(record.get("Human_info"), "Makeup focus areas")

            for text in self._knowledge_texts(record):
                document_id = knowledge_id(text)
                if document_id in seen:
                    # 첫 번째로 만난 쪽의 메타데이터를 쓴다. 같은 근거가 다른
                    # 피부 유형의 QA에도 실려 있을 수 있다. 한계는 D-027에 적었다.
                    continue
                seen.add(document_id)
                yield Document(
                    id=document_id,
                    text=text,
                    skin_type=skin_type,
                    area=area,
                    skin_detail=skin_detail,
                )


class AihubQaPairLoader(_QaReader):
    """QA쌍 하나를 Document 하나로 만든다 (M3-6, MEMORY.md D-032).

    본문은 "질문 + 답변"이다. 사용자의 새 질문이 비슷한 과거 질문을 찾고, 그
    질문에 달린 답변이 근거가 된다. FAQ 검색에 가까운 구조다.

    근거 발췌 색인(AihubQaLoader)과 비교해 F1이 0.238에서 0.315로 올랐다.
    대신 거부율이 1%에서 10%로 늘었다.

    ID 규칙은 AihubQaSetLoader와 같다. 평가할 때 색인에 든 케이스를 빼려면 두
    로더가 같은 ID를 내야 한다.
    """

    def load(self) -> Iterator[Document]:
        files = self._files()
        if not files:
            raise ValueError(f"QA 파일이 없다: {self._root}")

        used_ids: set[str] = set()
        for path in files:
            record = self._read(path)
            if record is None:
                continue

            annotation = record.get("Annotation_info")
            question = _text(annotation, "User Question")
            answer = _text(annotation, "Makeup Response")
            if not question or not answer:
                # 답변이 없으면 색인에 넣을 근거가 없다. 질문만으로는 못 쓴다.
                self.errors.append(f"{path.name}: 질문 또는 답변이 비어 있다")
                continue

            skin_type, skin_detail = _labels_from_folder(path.parent.name)
            yield Document(
                id=_unique(_seq_of(record) or path.stem, used_ids),
                text=f"질문: {question}\n답변: {answer}",
                skin_type=skin_type,
                area=_text(record.get("Human_info"), "Makeup focus areas"),
                skin_detail=skin_detail,
            )


class AihubQaSetLoader(_QaReader):
    """QA 라벨링을 평가 입력(EvalCase)으로 바꾼다 (M2-5).

    형식은 D-025에서 정한 것과 같다. 중간 JSON 파일을 만들지 않고 바로 EvalCase를
    만든다. 질문과 답변을 저장소에 다시 쓰면 원본 재배포에 해당한다 (D-006).
    """

    def load(self) -> list[EvalCase]:
        files = self._files()
        if not files:
            raise ValueError(f"QA 파일이 없다: {self._root}")

        cases: list[EvalCase] = []
        used_ids: set[str] = set()

        for path in files:
            record = self._read(path)
            if record is None:
                continue

            annotation = record.get("Annotation_info")
            question = _text(annotation, "User Question")
            if not question:
                # 질문이 없으면 평가에 쓸 수 없다. EvalCase가 생성 시점에 막는다.
                self.errors.append(f"{path.name}: 질문이 비어 있다")
                continue

            cases.append(
                EvalCase(
                    id=_unique(_seq_of(record) or path.stem, used_ids),
                    question=question,
                    expected=_text(annotation, "Makeup Response") or "",
                    gold_doc_ids=tuple(
                        knowledge_id(text) for text in self._knowledge_texts(record)
                    ),
                )
            )

        return cases


def _labels_from_folder(name: str) -> tuple[SkinType | None, str | None]:
    """`TL_민감성 피부_아토피 피부` → (SkinType.SENSITIVE, "아토피 피부").

    분류에 없는 표기가 나오면 SkinType은 None으로 두고 세부 유형만 남긴다.
    예외를 던지면 표기 하나 때문에 적재 전체가 멈춘다 (D-028).
    """
    parts = name.split("_")
    if len(parts) < 3:
        return None, None
    return CATEGORY_TO_SKIN_TYPE.get(parts[1]), "_".join(parts[2:]) or None


def _seq_of(record: dict[str, Any]) -> str:
    return _text(record.get("Data_info"), "SEQ") or ""


def _text(section: Any, key: str) -> str | None:
    """중첩 사전에서 문자열 값을 꺼낸다. 없거나 빈 값이면 None."""
    if not isinstance(section, dict):
        return None
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _unique(candidate: str, used: set[str]) -> str:
    """이미 쓴 ID면 번호를 붙인다.

    SEQ가 겹치는 파일이 있으면 케이스별로 결과를 되짚을 수 없다.
    """
    identifier = candidate
    suffix = 2
    while identifier in used:
        identifier = f"{candidate}-{suffix}"
        suffix += 1
    used.add(identifier)
    return identifier
