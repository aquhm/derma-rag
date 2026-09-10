"""OllamaGenerator 테스트.

가짜 세션으로 돌린다. 실제 Ollama가 필요한 것만 integration으로 표시한다.
"""

from __future__ import annotations

import pytest
import requests

from src.infrastructure.errors import GenerationUnavailable
from src.infrastructure.generation import DEFAULT_TIMEOUT, OllamaGenerator


class FakeResponse:
    def __init__(self, payload: object, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        if not self._responses:
            raise AssertionError("준비된 응답보다 요청이 많다")
        return self._responses.pop(0)


def reply(text: str) -> FakeResponse:
    return FakeResponse({"message": {"role": "assistant", "content": text}})


def test_returns_message_content() -> None:
    session = FakeSession([reply("자료에는 다음과 같이 기술돼 있습니다.")])

    result = OllamaGenerator(session=session).generate("프롬프트")

    assert result == "자료에는 다음과 같이 기술돼 있습니다."


def test_sends_prompt_as_a_single_user_message() -> None:
    session = FakeSession([reply("답")])

    OllamaGenerator(model="qwen2.5:3b", session=session).generate("프롬프트")

    sent = session.requests[0]
    assert sent["url"].endswith("/api/chat")
    assert sent["json"]["model"] == "qwen2.5:3b"
    assert sent["json"]["messages"] == [{"role": "user", "content": "프롬프트"}]
    assert sent["json"]["stream"] is False
    assert sent["timeout"] == DEFAULT_TIMEOUT


def test_temperature_is_zero_by_default() -> None:
    # 같은 질문에 같은 답이 나와야 평가 결과를 비교할 수 있다 (SRS NFR-4 재현성).
    session = FakeSession([reply("답")])

    OllamaGenerator(session=session).generate("프롬프트")

    assert session.requests[0]["json"]["options"]["temperature"] == 0.0


def test_context_size_is_sent_with_every_request() -> None:
    # Ollama 기본값 4,096을 넘으면 프롬프트 앞머리가 조용히 잘린다. 앞머리에
    # "진단하지 말라"는 지시가 있다 (SAFETY.md S-1).
    session = FakeSession([reply("답")])

    OllamaGenerator(session=session).generate("프롬프트")

    assert session.requests[0]["json"]["options"]["num_ctx"] == 8192


def test_context_size_can_be_changed() -> None:
    session = FakeSession([reply("답")])

    OllamaGenerator(num_ctx=16384, session=session).generate("프롬프트")

    assert session.requests[0]["json"]["options"]["num_ctx"] == 16384


def test_too_small_context_is_rejected() -> None:
    # 512 토큰이면 지시만으로 차고 근거가 들어가지 않는다.
    with pytest.raises(ValueError, match="num_ctx"):
        OllamaGenerator(num_ctx=128)


def test_blank_prompt_is_rejected() -> None:
    session = FakeSession([])

    with pytest.raises(ValueError, match="프롬프트"):
        OllamaGenerator(session=session).generate("   ")

    assert session.requests == []


def test_request_failure_becomes_generation_unavailable() -> None:
    session = FakeSession([FakeResponse({}, error=requests.HTTPError("500"))])

    with pytest.raises(GenerationUnavailable, match="생성 요청 실패"):
        OllamaGenerator(session=session).generate("프롬프트")


def test_error_message_does_not_carry_the_prompt() -> None:
    # 프롬프트에는 근거 문단과 사용자 질문이 들어 있다. 예외 메시지로 새면 로그에
    # 남는다 (SAFETY.md S-6, S-7).
    prompt = "민감성 피부가 당길 때 무엇을 주의해야 하나요?"
    session = FakeSession([FakeResponse({}, error=requests.HTTPError("500"))])

    with pytest.raises(GenerationUnavailable) as caught:
        OllamaGenerator(session=session).generate(prompt)

    assert prompt not in str(caught.value)


def test_non_json_response_is_rejected() -> None:
    session = FakeSession([FakeResponse({}, error=None)])
    session._responses[0]._payload = "JSON이 아님"  # type: ignore[attr-defined]

    with pytest.raises(GenerationUnavailable, match="응답 형식"):
        OllamaGenerator(session=session).generate("프롬프트")


def test_missing_content_is_rejected() -> None:
    session = FakeSession([FakeResponse({"message": {"role": "assistant"}})])

    with pytest.raises(GenerationUnavailable, match="응답 형식"):
        OllamaGenerator(session=session).generate("프롬프트")


@pytest.mark.parametrize("timeout", [0, -1])
def test_invalid_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        OllamaGenerator(timeout=timeout)


@pytest.mark.integration
def test_real_model_answers_in_korean() -> None:
    """실제 Ollama로 생성 1건. 모델은 qwen2.5:3b다 (MEMORY.md D-005)."""
    result = OllamaGenerator().generate(
        "다음 문장을 한국어 한 문장으로 요약하십시오: 민감성 피부는 자극에 빠르게 반응한다."
    )

    assert result.strip()
