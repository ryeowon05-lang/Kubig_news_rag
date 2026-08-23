from types import SimpleNamespace

import pytest

from generation.llm import LLMResult
from generation.openai_llm import OpenAIClient


class FakeResponses:
    def __init__(self, output_text='{"ok":true}'):
        self.output_text = output_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=self.output_text,
            model="gpt-5.4-mini-actual",
            usage=SimpleNamespace(input_tokens=11, output_tokens=4),
        )


class FakeClient:
    def __init__(self, output_text='{"ok":true}'):
        self.responses = FakeResponses(output_text)


def test_openai_client_uses_responses_structured_output_and_metadata():
    fake = FakeClient()
    client = OpenAIClient(client=fake, model="test-model", max_tokens=321)
    assert client.provider == "openai"
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    result = client.generate(
        "system", "user", temperature=0, output_schema=schema, return_metadata=True,
    )

    assert result == LLMResult('{"ok":true}', "gpt-5.4-mini-actual", 11, 4)
    call = fake.responses.calls[0]
    assert call["model"] == "test-model"
    assert call["instructions"] == "system"
    assert call["input"] == "user"
    assert call["max_output_tokens"] == 321
    assert call["temperature"] == 0
    assert call["store"] is False
    assert call["text"]["format"] == {
        "type": "json_schema", "name": "structured_output", "strict": True, "schema": schema,
    }


def test_openai_client_keeps_plain_string_default_behavior():
    fake = FakeClient("plain answer")
    result = OpenAIClient(client=fake).generate("system", "user")
    assert result == "plain answer"
    assert "temperature" not in fake.responses.calls[0]
    assert "text" not in fake.responses.calls[0]


def test_openai_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIClient()


def test_openai_client_rejects_empty_output():
    with pytest.raises(RuntimeError, match="empty"):
        OpenAIClient(client=FakeClient("")).generate("system", "user")
