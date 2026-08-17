from types import SimpleNamespace

import pytest

from generation.llm import ClaudeClient, LLMError
from generation.prompt import SYSTEM_PROMPT, build_messages, build_user_prompt
from rag import RAGPipeline


RESULTS = [
    {
        "chunk_id": "news-1",
        "title": "삼성전자, 반도체 실적 개선 기대",
        "date": "2026.08.08",
        "url": "https://example.com/1",
        "stock_names": ["삼성전자"],
        "content": "삼성전자의 반도체 사업 실적 개선이 기대된다는 분석이 나왔다.",
        "similarity": 0.91,
    },
    {
        "chunk_id": "news-2",
        "title": "반도체 업황 회복 전망",
        "date": "2026.08.07",
        "url": "https://example.com/2",
        "stock_names": ["삼성전자", "SK하이닉스"],
        "content": "메모리 업황이 회복될 가능성이 있다는 전망이 제기됐다.",
        "similarity": 0.84,
    },
]


def test_prompt_contains_question_and_retrieved_news():
    prompt = build_user_prompt("삼성전자 반도체 실적 전망은?", RESULTS)
    assert "삼성전자 반도체 실적 전망은?" in prompt
    assert "[뉴스1]" in prompt
    assert "[뉴스2]" in prompt
    assert RESULTS[0]["content"] in prompt
    assert RESULTS[0]["url"] in prompt


def test_prompt_requires_grounded_answer():
    system, user = build_messages("질문", RESULTS)
    assert "제공된 자료에서 확인할 수 없습니다." in system
    assert "뉴스 자료에 직접 근거가 있는 내용만" in user


def test_prompt_rejects_empty_question_or_results():
    with pytest.raises(ValueError):
        build_user_prompt("   ", RESULTS)
    with pytest.raises(ValueError):
        build_user_prompt("질문", [])


class FakeMessages:
    def __init__(self, answer="테스트 답변 [뉴스1]"):
        self.answer = answer
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.answer)]
        )


class FakeClient:
    def __init__(self, answer="테스트 답변 [뉴스1]"):
        self.messages = FakeMessages(answer)


def test_claude_client_sends_expected_messages():
    fake = FakeClient()
    client = ClaudeClient(client=fake, model="test-model", max_tokens=123)

    answer = client.generate("system", "user")

    assert answer == "테스트 답변 [뉴스1]"
    call = fake.messages.calls[0]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == 123
    assert call["system"] == "system"
    assert call["messages"] == [{"role": "user", "content": "user"}]


def test_claude_client_raises_on_empty_response():
    fake = FakeClient(answer="")
    client = ClaudeClient(client=fake)
    with pytest.raises(LLMError, match="empty"):
        client.generate("system", "user")


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.results


class FakeLLM:
    def __init__(self, answer="삼성전자의 반도체 실적 개선이 기대됩니다. [뉴스1]"):
        self.answer = answer
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.answer


def test_rag_pipeline_connects_retrieval_prompt_and_generation():
    retriever = FakeRetriever(RESULTS)
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = pipeline.ask("삼전 반도체 실적 전망", top_k=5)

    assert response.answer.startswith("삼성전자의")
    assert retriever.calls == [("삼성전자 반도체 실적 전망", 5)]
    assert len(llm.calls) == 1
    assert "삼성전자 반도체 실적 전망" in llm.calls[0][1]
    assert response.sources[0]["news_number"] == 1
    assert response.sources[0]["title"] == RESULTS[0]["title"]
    assert response.sources[0]["url"] == RESULTS[0]["url"]


def test_rag_pipeline_returns_no_answer_when_retrieval_is_empty():
    retriever = FakeRetriever([])
    llm = FakeLLM()
    pipeline = RAGPipeline(retriever=retriever, llm=llm)

    response = pipeline.ask("삼성전자 전망")

    assert response.answer == "제공된 자료에서 확인할 수 없습니다."
    assert response.sources == []
    assert llm.calls == []
