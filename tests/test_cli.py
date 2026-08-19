"""CLI that exposes every retrieval technique the Streamlit app has."""

import json
from argparse import Namespace

import pytest

import cli
from indexing.expanding import ExpandingRetriever
from indexing.hybrid import HybridRetriever


class FakeRetriever:
    def __init__(self, name, results=None):
        self.name = name
        self.results = results or []
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.results


class FakeLLM:
    """Records calls. Never reaches the network."""

    def __init__(self):
        self.generate_calls = []
        self.stream_calls = []

    def generate(self, system_prompt, user_prompt, history=None):
        self.generate_calls.append((system_prompt, user_prompt, history))
        return "답변"

    def stream(self, system_prompt, user_prompt, history=None):
        self.stream_calls.append((system_prompt, user_prompt, history))
        return iter(["답", "변"])


class FakeLoaders:
    def __init__(self, results=None):
        self.dense_retriever = FakeRetriever("dense", results)
        self.bm25_retriever = FakeRetriever("bm25", results)
        self.claude = FakeLLM()

    def dense(self):
        return self.dense_retriever

    def bm25(self):
        return self.bm25_retriever

    def llm(self):
        return self.claude


# --- 인자 파싱 -------------------------------------------------------------


def test_the_default_run_is_hybrid_without_expansion_or_gate():
    parsed = cli.parse_args(["삼성전자 실적 어때?"])

    assert parsed.question == "삼성전자 실적 어때?"
    assert parsed.mode == "hybrid"
    assert parsed.expand == "none"
    assert parsed.gate is False
    assert parsed.top_k == 5


def test_every_retrieval_technique_is_reachable_from_flags():
    parsed = cli.parse_args(
        ["질문", "--mode", "bm25", "--expand", "hyde", "--gate", "--top-k", "3"]
    )

    assert parsed.mode == "bm25"
    assert parsed.expand == "hyde"
    assert parsed.gate is True
    assert parsed.top_k == 3


def test_chat_mode_takes_no_question_on_the_command_line():
    parsed = cli.parse_args(["--chat"])

    assert parsed.chat is True
    assert parsed.question is None


def test_a_question_is_required_outside_chat_mode():
    with pytest.raises(SystemExit):
        cli.parse_args([])


# --- 리트리버 조립 ---------------------------------------------------------


def test_dense_mode_searches_with_embeddings_only():
    loaders = FakeLoaders()

    retriever = cli.build_retriever(cli.parse_args(["q", "--mode", "dense"]), loaders)

    assert retriever is loaders.dense_retriever


def test_bm25_mode_searches_with_tokens_only():
    loaders = FakeLoaders()

    retriever = cli.build_retriever(cli.parse_args(["q", "--mode", "bm25"]), loaders)

    assert retriever is loaders.bm25_retriever


def test_hybrid_mode_fuses_both_retrievers():
    loaders = FakeLoaders()

    retriever = cli.build_retriever(cli.parse_args(["q", "--mode", "hybrid"]), loaders)

    assert isinstance(retriever, HybridRetriever)
    assert retriever._dense is loaders.dense_retriever
    assert retriever._bm25 is loaders.bm25_retriever


def test_expansion_wraps_whatever_retriever_was_chosen():
    loaders = FakeLoaders()

    retriever = cli.build_retriever(
        cli.parse_args(["q", "--mode", "dense", "--expand", "hyde"]), loaders
    )

    assert isinstance(retriever, ExpandingRetriever)
    assert retriever.base is loaders.dense_retriever


def test_no_expansion_leaves_the_retriever_unwrapped():
    loaders = FakeLoaders()

    retriever = cli.build_retriever(cli.parse_args(["q", "--expand", "none"]), loaders)

    assert not isinstance(retriever, ExpandingRetriever)


def test_rrf_settings_reach_the_hybrid_retriever():
    loaders = FakeLoaders()

    retriever = cli.build_retriever(
        cli.parse_args(["q", "--candidate-k", "30", "--rrf-k", "10"]), loaders
    )

    assert retriever.candidate_k == 30
    assert retriever.rrf_k == 10


# --- 파이프라인 조립 -------------------------------------------------------


def test_the_gate_is_off_unless_asked_for():
    pipeline = cli.build_pipeline(cli.parse_args(["q"]), FakeLoaders())

    assert pipeline.gate is None


def test_the_gate_flag_lets_small_talk_skip_retrieval():
    pipeline = cli.build_pipeline(cli.parse_args(["q", "--gate"]), FakeLoaders())

    assert pipeline.gate is not None


# --- 출력 ------------------------------------------------------------------


def test_a_source_shows_its_title_link_and_provenance():
    text = cli.format_source(
        1,
        {
            "title": "삼성전자 실적 개선",
            "url": "https://example.com/a",
            "date": "2026.08.08",
            "rrf_score": 0.0164,
            "dense_rank": 1,
            "bm25_rank": None,
        },
    )

    assert "삼성전자 실적 개선" in text
    assert "https://example.com/a" in text
    assert "RRF 0.0164" in text
    assert "dense #1" in text


def test_a_source_without_a_title_still_renders():
    assert "(제목 없음)" in cli.format_source(1, {})


def test_retrieval_only_prints_sources_and_never_calls_the_llm(capsys):
    loaders = FakeLoaders(results=[{"title": "기사 하나", "url": "https://ex.com/1"}])

    cli.main(["질문", "--mode", "dense", "--retrieval-only"], loaders=loaders)

    assert "기사 하나" in capsys.readouterr().out
    assert loaders.claude.generate_calls == []
    assert loaders.claude.stream_calls == []


def test_the_answer_is_printed_as_it_streams(capsys):
    loaders = FakeLoaders(results=[{"title": "기사", "url": "https://ex.com/1"}])

    cli.main(["질문", "--mode", "dense"], loaders=loaders)

    assert "답변" in capsys.readouterr().out
    assert len(loaders.claude.stream_calls) == 1


def test_json_output_is_machine_readable(capsys):
    loaders = FakeLoaders(results=[{"title": "기사", "url": "https://ex.com/1"}])

    cli.main(["질문", "--mode", "dense", "--json"], loaders=loaders)

    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == "답변"
    assert payload["sources"][0]["title"] == "기사"


# --- 대화 모드 -------------------------------------------------------------


class RecordingPipeline:
    def __init__(self):
        self.histories = []

    def ask_stream(self, question, top_k=5, history=None):
        from rag import StreamingRAGResponse

        self.histories.append(list(history or []))
        return StreamingRAGResponse(
            question=question, sources=[], searched=True, stream=iter(["응답"])
        )


def test_chat_mode_carries_the_previous_turn_into_the_next_question():
    pipeline = RecordingPipeline()
    turns = iter(["첫 질문", "그럼 SK하이닉스는?", "exit"])

    cli.run_chat(pipeline, Namespace(top_k=5, json=False), input_fn=lambda _: next(turns))

    assert pipeline.histories[0] == []
    assert pipeline.histories[1] == [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "응답"},
    ]


def test_chat_mode_stops_at_end_of_input():
    pipeline = RecordingPipeline()

    def eof(_):
        raise EOFError

    cli.run_chat(pipeline, Namespace(top_k=5, json=False), input_fn=eof)

    assert pipeline.histories == []


# --- 실패 처리 -------------------------------------------------------------


def test_retrieval_only_cannot_be_combined_with_chat():
    with pytest.raises(SystemExit):
        cli.parse_args(["--chat", "--retrieval-only"])


def test_a_missing_index_is_reported_instead_of_a_traceback(capsys):
    class BrokenLoaders(FakeLoaders):
        def dense(self):
            raise FileNotFoundError("chroma db not found: data/chroma")

    exit_code = cli.main(["질문", "--mode", "dense"], loaders=BrokenLoaders())

    assert exit_code == 1
    assert "chroma db not found" in capsys.readouterr().err
