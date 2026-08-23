import json
import time

import pytest

from evaluation.adjudication import CostBudgetExceeded, CostTracker, judge_packets
from generation.llm import LLMResult


def packets():
    return [{
        "query_id": "K001", "question": "투자 목적은?", "category": "factoid",
        "candidates": [
            {"candidate_key": f"opaque-{i}", "title": f"기사 {i}", "date": "2026.08.01",
             "content": f"본문 {i}", "url": "u", "doc_type": "article"}
            for i in range(4)
        ],
    }]


class LocalJudge:
    provider = "openai"
    model = "gpt-5.6-luna"
    max_tokens = 128

    def __init__(self):
        self.calls = []

    def generate(self, system, user, **options):
        payload = json.loads(user)
        self.calls.append((payload, options))
        wrapped = payload["candidate"]["candidate_data"]
        candidate = json.loads(wrapped.splitlines()[1])
        time.sleep((3 - int(candidate["content"].split()[-1])) * .002)
        return LLMResult(
            json.dumps({"label": "relevant", "rationale": "질문의 답을 명시한다."}, ensure_ascii=False),
            "gpt-5.6-luna", 100, 10, 20, 3,
        )


def test_candidate_local_schema_local_key_binding_and_original_order(tmp_path):
    judge = LocalJudge()
    rows, stats = judge_packets(
        packets(), judge, checkpoint_dir=tmp_path, phase="pass1", concurrency=4,
        cost_tracker=CostTracker(1.20), sleep=lambda _: None,
    )
    assert [row["candidate_key"] for row in rows] == [f"opaque-{i}" for i in range(4)]
    assert all("candidate_key" not in call[0] for call in judge.calls)
    assert all("candidate_key" not in call[1]["output_schema"]["properties"] for call in judge.calls)
    assert all("temperature" not in call[1] for call in judge.calls)
    assert all(call[1]["reasoning_effort"] == "low" for call in judge.calls)
    assert stats["input_tokens"] == 400 and stats["cached_input_tokens"] == 80
    assert stats["output_tokens"] == 40 and stats["reasoning_tokens"] == 12


def test_non_openai_judge_keeps_existing_temperature_behavior(tmp_path):
    judge = LocalJudge()
    judge.provider = "anthropic"
    judge.model = "claude-sonnet-5"
    judge_packets(packets()[:1], judge, checkpoint_dir=tmp_path, phase="pass1",
                  concurrency=1, cost_tracker=CostTracker(1.20), sleep=lambda _: None)
    assert all(call[1]["temperature"] == 0 for call in judge.calls)


def test_openai_response_model_mismatch_stops_without_checkpoint(tmp_path):
    judge = LocalJudge()
    original = judge.generate

    def mismatched(*args, **kwargs):
        result = original(*args, **kwargs)
        return LLMResult(result.text, "different-model", result.input_tokens,
                         result.output_tokens, result.cached_input_tokens, result.reasoning_tokens)

    judge.generate = mismatched
    with pytest.raises(RuntimeError, match="response model differs"):
        judge_packets(packets()[:1], judge, checkpoint_dir=tmp_path, phase="pass1",
                      concurrency=1, cost_tracker=CostTracker(1.20), sleep=lambda _: None)
    assert not (tmp_path / "pass1.json").exists()


def test_atomic_checkpoint_resume_and_phase_separation(tmp_path):
    first = LocalJudge()
    rows, _ = judge_packets(packets(), first, checkpoint_dir=tmp_path, phase="pass1",
                            concurrency=4, cost_tracker=CostTracker(1.20), sleep=lambda _: None)
    assert len(rows) == 4
    assert len(json.loads((tmp_path / "pass1.json").read_text(encoding="utf-8"))) == 4
    second = LocalJudge()
    resumed, stats = judge_packets(packets(), second, checkpoint_dir=tmp_path, phase="pass1",
                                   concurrency=4, cost_tracker=CostTracker(1.20), sleep=lambda _: None)
    assert resumed == rows and not second.calls and stats["checkpoint_hits"] == 4
    judge_packets(packets(), LocalJudge(), checkpoint_dir=tmp_path, phase="review",
                  concurrency=4, cost_tracker=CostTracker(1.20), sleep=lambda _: None)
    assert (tmp_path / "review.json").is_file()


def test_invalid_label_creates_no_fallback(tmp_path):
    judge = LocalJudge()
    judge.generate = lambda *args, **kwargs: LLMResult(
        json.dumps({"label": "maybe", "rationale": "x"}), "gpt-5.6-luna", 1, 1, 0, 0)
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        judge_packets(packets(), judge, checkpoint_dir=tmp_path, phase="pass1", concurrency=1,
                      cost_tracker=CostTracker(1.20), sleep=lambda _: None)
    assert not (tmp_path / "pass1.json").exists()


def test_cost_cap_stops_before_new_request(tmp_path):
    judge = LocalJudge()
    with pytest.raises(CostBudgetExceeded):
        judge_packets(packets(), judge, checkpoint_dir=tmp_path, phase="pass1", concurrency=1,
                      cost_tracker=CostTracker(0.000001), sleep=lambda _: None)
    assert not judge.calls
