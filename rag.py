"""End-to-end dense-retrieval + Claude generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from generation.llm import ClaudeClient
from generation.prompt import build_messages
from indexing.query_preprocess import normalize_query
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RAGResponse:
    question: str
    answer: str
    sources: list[dict]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
        }


class RAGPipeline:
    """Question -> retrieval -> grounded Claude answer + retrieved sources."""

    def __init__(self, retriever: Any | None = None, llm: Any | None = None) -> None:
        if retriever is None:
            from indexing.retriever import get_retriever
            retriever = get_retriever()
        self.retriever = retriever
        self.llm = llm or ClaudeClient()

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K) -> RAGResponse:
        question = question.strip()
        if not question:
            raise ValueError("question cannot be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        question = normalize_query(question)
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            return RAGResponse(
                question=question,
                answer="제공된 자료에서 확인할 수 없습니다.",
                sources=[],
            )

        system_prompt, user_prompt = build_messages(question, results)
        answer = self.llm.generate(system_prompt, user_prompt)
        return RAGResponse(
            question=question,
            answer=answer,
            sources=[self._source_from_result(rank, result) for rank, result in enumerate(results, 1)],
        )

    @staticmethod
    def _source_from_result(rank: int, result: dict) -> dict:
        return {
            "news_number": rank,
            "title": result.get("title", ""),
            "date": result.get("date", ""),
            "url": result.get("url", ""),
            "stock_names": result.get("stock_names", []),
            "similarity": result.get("similarity"),
        }


def ask(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Convenience function for one-shot use from another script."""
    return RAGPipeline().ask(question, top_k=top_k).to_dict()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Korean financial-news question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    response = ask(args.question, top_k=args.top_k)
    print(json.dumps(response, ensure_ascii=False, indent=2))
