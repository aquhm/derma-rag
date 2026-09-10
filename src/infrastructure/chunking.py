"""Chunker 어댑터.

빈 줄로 문단을 나누고, 문단이 최대 길이를 넘으면 문장 단위로 다시 나눈다.

기본값 512자와 오버랩 64자는 docs/guidelines/03-infrastructure.md의 초기값이며
**가설이다.** M3-1에서 Recall@5로 측정해 조정한다. 그래서 값을 상수로 박지 않고
생성자 인자로 받는다. Strategy 패턴을 쓰는 이유도 같다. 다른 청킹 방식을 만들어
나란히 비교할 것이다 (docs/ARCHITECTURE.md 6절).
"""

from __future__ import annotations

import re

from src.domain.models import Chunk, Document

# 빈 줄 하나 이상을 문단 경계로 본다.
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

# 문장 끝 부호 뒤의 공백에서 나눈다.
# (?<![0-9])는 "1. 유분을 먼저 정리한다" 같은 번호 목록에서 잘리는 것을 막는다.
# 숫자 뒤의 마침표는 문장 끝이 아니라 항목 번호일 때가 많다.
SENTENCE_BREAK = re.compile(r"(?<![0-9])(?<=[.!?])\s+")


class ParagraphChunker:
    """application.ports.Chunker를 만족한다."""

    def __init__(self, max_chars: int = 512, overlap: int = 64) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars는 1 이상이어야 한다: {max_chars}")
        if overlap < 0:
            raise ValueError(f"overlap은 0 이상이어야 한다: {overlap}")
        if overlap >= max_chars:
            # 오버랩이 최대 길이 이상이면 다음 청크가 앞 청크를 통째로 물고 시작해
            # 진도가 나가지 않는다.
            raise ValueError(f"overlap({overlap})은 max_chars({max_chars})보다 작아야 한다")

        self._max_chars = max_chars
        self._overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        texts: list[str] = []

        for paragraph in PARAGRAPH_BREAK.split(doc.text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= self._max_chars:
                texts.append(paragraph)
            else:
                texts.extend(self._split_paragraph(paragraph))

        # 청크 ID에 순번을 담는다. Chunk에 위치 필드를 따로 두지 않고 "doc-1#0"
        # 형식으로 순서를 표현한다.
        return [
            Chunk(
                id=f"{doc.id}#{position}",
                doc_id=doc.id,
                text=text,
                skin_type=doc.skin_type,
                area=doc.area,
                skin_detail=doc.skin_detail,
            )
            for position, text in enumerate(texts)
        ]

    def _split_paragraph(self, paragraph: str) -> list[str]:
        """문장을 최대 길이까지 채워 묶는다. 묶음 사이에는 오버랩을 넣는다."""
        groups: list[str] = []
        current = ""

        for sentence in self._sentences(paragraph):
            # 문장 하나가 이미 최대 길이를 넘으면 더 나눌 기준이 없다. 이 문장만
            # 문자 수로 잘라 그대로 내보낸다. 다른 문장과 묶지 않는 이유는, 묶는
            # 과정에서 조각 사이에 원문에 없던 공백이 끼어들기 때문이다.
            if len(sentence) > self._max_chars:
                if current:
                    groups.append(current)
                    current = ""
                groups.extend(self._windows(sentence))
                continue

            if not current:
                current = sentence
                continue

            if len(current) + 1 + len(sentence) <= self._max_chars:
                current = f"{current} {sentence}"
                continue

            groups.append(current)
            tail = self._tail_for(previous=current, next_sentence=sentence)
            current = f"{tail} {sentence}" if tail else sentence

        if current:
            groups.append(current)
        return groups

    def _sentences(self, paragraph: str) -> list[str]:
        return [s.strip() for s in SENTENCE_BREAK.split(paragraph) if s.strip()]

    def _windows(self, text: str) -> list[str]:
        """긴 문장을 겹치는 창으로 자른다. 각 조각은 원문의 부분 문자열 그대로다."""
        step = self._max_chars - self._overlap  # 생성자 검사로 1 이상이 보장된다
        windows: list[str] = []
        start = 0

        while start < len(text):
            windows.append(text[start : start + self._max_chars])
            if start + self._max_chars >= len(text):
                break
            start += step

        return windows

    def _tail_for(self, previous: str, next_sentence: str) -> str:
        """다음 묶음 앞에 붙일 오버랩. 붙여도 최대 길이를 넘지 않는 만큼만 준다."""
        if self._overlap == 0:
            return ""
        room = self._max_chars - len(next_sentence) - 1
        size = min(self._overlap, room, len(previous))
        return previous[-size:] if size > 0 else ""


class WholeDocumentChunker:
    """문서 하나를 청크 하나로 그대로 내보낸다.

    QA쌍 색인에 쓴다 (D-032). QA 텍스트는 평균 855자라 512자 청커에 걸려 2~3
    조각으로 쪼개졌고, 그러면 상위 k를 같은 QA의 조각들이 차지한다. 실측에서
    근거 3건 중 2건이 같은 QA였다.

    application.ports.Chunker를 만족한다. 청킹을 Strategy로 둔 이유가 이것이다
    (docs/ARCHITECTURE.md 6절).
    """

    def split(self, doc: Document) -> list[Chunk]:
        return [
            Chunk(
                id=f"{doc.id}#0",
                doc_id=doc.id,
                text=doc.text,
                skin_type=doc.skin_type,
                area=doc.area,
                skin_detail=doc.skin_detail,
            )
        ]
