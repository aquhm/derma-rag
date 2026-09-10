"""Generator 어댑터.

Ollama의 POST /api/chat을 쓴다. 프롬프트는 application이 이미 조립해서 넘긴다.
여기서 하는 일은 그 문자열을 모델이 받는 요청 형식으로 바꾸는 것뿐이다.

모델은 qwen2.5:3b다. gemma4:31b는 응답 1건에 189초가 걸리고 지시를 따르지 않아
교체했다 (MEMORY.md D-005, ERRORS.md E-001).

타임아웃이 임베딩(30초)보다 훨씬 긴 이유는 생성이 토큰 수에 비례해 오래 걸리기
때문이다. 권장값은 docs/guidelines/03-infrastructure.md에 있다.
"""

from __future__ import annotations

import requests

from src.infrastructure.errors import GenerationUnavailable

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT = 300.0

# 같은 질문에 같은 답이 나와야 평가 점수를 비교할 수 있다. 기준선(M2-8) 대비로
# 개선을 판단하려면 생성이 흔들리면 안 된다.
DEFAULT_TEMPERATURE = 0.0

# 컨텍스트 크기를 명시한다. Ollama 기본값은 4,096 토큰이고, 넘치면 **경고 없이
# 앞부분을 버린다.** 먼저 잘리는 것은 프롬프트 앞머리, 즉 "진단하지 말라"는
# 지시다. 안전 지시가 조용히 사라지는 실패는 눈에 띄지 않는다.
#
# 지금까지는 문제가 없었다. QA쌍 근거 3개면 약 2,100 토큰이다. 그러나 M3-7의
# 논문 청크(2,048자)는 3개만 실어도 4,096에 근접한다. 재료를 바꿀 때마다 이
# 한계를 손으로 다시 계산하게 두는 것은 E-004를 반복하는 길이다.
DEFAULT_NUM_CTX = 8192


class OllamaGenerator:
    """application.ports.Generator를 만족한다."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        num_ctx: int = DEFAULT_NUM_CTX,
        session: requests.Session | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout은 0보다 커야 한다: {timeout}")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"temperature는 [0, 2] 안이어야 한다: {temperature}")
        if num_ctx < 512:
            raise ValueError(f"num_ctx는 512 이상이어야 한다: {num_ctx}")

        self._model = model
        self._url = f"{host.rstrip('/')}/api/chat"
        self._timeout = timeout
        self._temperature = temperature
        self._num_ctx = num_ctx
        self._session = session if session is not None else requests.Session()

    def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("빈 프롬프트로는 생성할 수 없다")

        try:
            response = self._session.post(
                self._url,
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    # 스트리밍을 끄면 응답이 JSON 한 덩어리로 온다. CLI는 완성된
                    # 답을 안전 검사에 넘겨야 하므로 조각으로 받을 이유가 없다.
                    "stream": False,
                    "options": {
                        "temperature": self._temperature,
                        "num_ctx": self._num_ctx,
                    },
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            # 예외 메시지에 프롬프트를 넣지 않는다. 근거 문단과 사용자 질문이
            # 들어 있다 (SAFETY.md S-6, S-7).
            raise GenerationUnavailable(
                f"생성 요청 실패 (model={self._model}): {exc}"
            ) from exc
        except ValueError as exc:
            raise GenerationUnavailable(
                f"생성 응답 형식이 아니다. JSON이 아니다 (model={self._model})"
            ) from exc

        return self._extract(payload)

    def _extract(self, payload: object) -> str:
        """응답에서 본문만 꺼낸다.

        형식을 검사하는 이유는 Ollama 버전이 바뀌면 응답 모양이 달라지기 때문이다.
        여기서 막지 않으면 빈 문자열이 답변 경로로 흘러간다.
        """
        if not isinstance(payload, dict):
            raise GenerationUnavailable(
                f"생성 응답 형식이 아니다. 사전이 아니다 (model={self._model})"
            )

        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(
            message.get("content"), str
        ):
            raise GenerationUnavailable(
                f"생성 응답 형식이 아니다. message.content가 없다 (model={self._model})"
            )

        return str(message["content"])
