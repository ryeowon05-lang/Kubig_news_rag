import hashlib

from evaluation.runner import run_evaluation


def hit(chunk_id, article_id):
    return {
        "chunk_id": chunk_id, "article_id": article_id, "title": f"기사 {article_id}",
        "date": "2026.08.01", "content": "본문", "url": f"https://example.com/{article_id}", "doc_type": "article",
    }


class FakeRetriever:
    def __init__(self, hits=None, error=None):
        self.hits = hits or []
        self.error = error
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        if self.error:
            raise RuntimeError(self.error)
        return self.hits[:top_k]


def record(method="source_seeded"):
    return {
        "query_id": "q1", "question": "질문", "construction_method": method,
        "category": "entity", "seed_article_id": 2 if method == "source_seeded" else None,
        "gold_article_ids": [2, 9], "evidence": [], "author": "a", "reviewer": "b",
        "review_status": "approved", "annotation_minutes": 10, "naturalness": 4,
        "gold_clarity": 5, "notes": "",
    }


def test_runner_creates_paired_raw_and_unique_article_results_and_hash(tmp_path):
    questions = tmp_path / "questions.jsonl"
    questions.write_bytes(b"question bytes\n")
    dense = FakeRetriever([hit("1a", 1), hit("1b", 1), hit("2", 2), hit("3", 3), hit("4", 4), hit("5", 5)])
    hybrid = FakeRetriever([hit("2", 2), hit("6", 6)])

    report = run_evaluation(
        [record()], {"dense": dense, "hybrid": hybrid}, question_path=questions,
        corpus_article_count=432, corpus_chunk_count=1377,
        settings={"candidate_k": 20, "rrf_k": 60, "model": "BAAI/bge-m3"},
        git_commit="abc123", chunk_depth=100, now=lambda: "2026-08-20T00:00:00+00:00",
    )

    assert dense.calls == [("질문", 100)] and hybrid.calls == [("질문", 100)]
    assert report["metadata"]["question_sha256"] == hashlib.sha256(b"question bytes\n").hexdigest()
    assert report["metadata"]["corpus"] == {"article_count": 432, "chunk_count": 1377}
    assert report["metadata"]["git_commit"] == "abc123"
    by_system = {result["system"]: result for result in report["results"]}
    assert by_system["dense"]["raw_chunk_ranking"][1] == {"rank": 2, "chunk_id": "1b", "article_id": 1}
    assert [item["article_id"] for item in by_system["dense"]["unique_article_ranking"]] == [1, 2, 3, 4, 5]
    assert by_system["dense"]["first_relevant_rank"] == 2
    assert by_system["hybrid"]["hit_at_1"] == 1
    assert by_system["dense"]["top_5_overlap"] == 1


def test_runner_records_one_system_error_without_losing_paired_result(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text("", encoding="utf-8")
    report = run_evaluation(
        [record()], {"dense": FakeRetriever(error="boom"), "hybrid": FakeRetriever([hit("2", 2)])},
        question_path=path, corpus_article_count=1, corpus_chunk_count=1, settings={}, git_commit="abc",
    )
    by_system = {result["system"]: result for result in report["results"]}
    assert by_system["dense"]["error"] == "RuntimeError: boom"
    assert by_system["hybrid"]["hit_at_1"] == 1


def test_runner_summarizes_each_construction_method():
    records = [record("source_seeded"), {**record("query_first"), "query_id": "q2", "evidence": [{"relevance": "uncertain"}]}]
    report = run_evaluation(
        records, {"dense": FakeRetriever([hit("2", 2)]), "hybrid": FakeRetriever([])},
        question_path=__file__, corpus_article_count=432, corpus_chunk_count=1, settings={}, git_commit="abc",
    )
    summary = {item["construction_method"]: item for item in report["summaries"]}
    assert summary["source_seeded"]["question_count"] == 1
    assert summary["source_seeded"]["mean_annotation_minutes"] == 10.0
    assert summary["source_seeded"]["gold_article_count"] == 2
    assert summary["query_first"]["uncertain_evidence_count"] == 1
    assert summary["source_seeded"]["systems"]["dense"]["hit_at_1"] == 1.0
    assert "unstable" in summary["source_seeded"]["warning"].lower()
