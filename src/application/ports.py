"""포트 정의. infrastructure가 구현할 계약이다.

포트를 application에 두는 이유는 의존을 역전시키기 위해서다. application은
infrastructure를 import 하지 않고도 그 기능을 쓴다. 포트를 infrastructure에 두면
이 방향이 무너진다 (docs/guidelines/02-application.md).

Protocol을 쓰는 이유는 상속이 필요 없기 때문이다. 어댑터가 이 클래스들을 상속하지
않아도 메서드 시그니처만 맞으면 성립한다.

@runtime_checkable을 붙이지 않는다. isinstance 검사가 가능해지지만 메서드 이름만
확인하고 시그니처는 확인하지 않는다. 인자가 틀려도 통과하므로 거짓 안심을 준다.
목을 쓰지 않는 이유와 같다 (docs/guidelines/05-testing.md).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from src.domain.models import Chunk, Document, Evidence


class DocumentLoader(Protocol):
    """1단계. 원본을 읽어 Document로 만든다.

    반환 타입이 list가 아니라 Iterable인 이유는 제너레이터를 허용하기 위해서다.
    실데이터는 9,377건이라 전부 메모리에 올리지 않고 흘려보낼 수 있어야 한다.
    """

    def load(self) -> Iterable[Document]: ...


class Chunker(Protocol):
    """2단계. 문서 하나를 청크 여러 개로 나눈다.

    문서 단위로 받는 이유는 청킹 전략을 갈아 끼우며 측정 비교할 것이기 때문이다
    (docs/ARCHITECTURE.md 6절 Strategy).
    """

    def split(self, doc: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    """3단계. 텍스트를 벡터로 만든다.

    배치 메서드 하나만 둔다. 질문 1건은 embed([query])[0]으로 쓴다. 단건 메서드를
    따로 두면 어댑터가 구현할 것이 둘로 늘어나고 둘의 동작이 어긋날 여지가 생긴다.

    반환이 list[list[float]]인 이유는 domain과 application이 numpy를 모르기
    때문이다. 배열 표현은 infrastructure의 사정이다 (docs/ARCHITECTURE.md 2절).
    """

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorIndex(Protocol):
    """4단계와 5단계. 벡터를 담고 검색한다.

    load()가 없는 이유는 반쯤 초기화된 색인 상태를 만들지 않기 위해서다. 파일에서
    읽는 것은 어댑터의 classmethod(NumpyVectorIndex.from_file)가 맡고, 그 호출은
    composition.py에서만 일어난다. 포트에 load를 두면 "비어 있는 색인에 search를
    부르면 무엇이 나오는가"라는 답할 필요 없는 질문이 생긴다 (MEMORY.md D-013).
    """

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[Evidence]: ...

    def save(self, path: Path) -> None: ...


class Generator(Protocol):
    """6단계. 조립된 프롬프트로 답 본문을 만든다.

    프롬프트 문자열은 application이 만든다. 이 포트는 모델이 무엇인지 모른다.
    모델별 요청 포맷 변환은 어댑터의 일이다.
    """

    def generate(self, prompt: str) -> str: ...
