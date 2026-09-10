"""어댑터가 던지는 예외.

외부 시스템 장애를 여기서 정의한다. domain에 두지 않는 이유는, 서비스가 죽은 것과
"근거가 없어서 답하지 않는다"(Refusal)가 다른 종류의 실패이기 때문이다. 둘을 같은
타입으로 표현하면 Ollama가 꺼져 있을 때 사용자에게 "관련 자료를 찾지 못했습니다"라고
잘못 안내하게 된다.

이 예외들은 application을 지나 interface(CLI)까지 올라간다. application은 이 예외를
알 필요가 없고, 알아서도 안 된다. import 방향이 뒤집힌다.

예외 메시지에 원본 텍스트를 넣지 않는다. 건수와 모델 이름만 넣는다
(docs/guidelines/00-common.md 로그 항목).
"""

from __future__ import annotations


class OllamaUnavailable(RuntimeError):
    """Ollama에 요청했으나 결과를 얻지 못했다."""


class EmbeddingUnavailable(OllamaUnavailable):
    """임베딩 요청이 실패했다."""


class GenerationUnavailable(OllamaUnavailable):
    """생성 요청이 실패했다."""
