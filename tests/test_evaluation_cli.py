import json
import subprocess
import sys

import pytest

from evaluation.cli import main


def article_file(path):
    path.write_text(json.dumps([{
        "id": 1, "title": "반도체", "content": "HBM", "date": "2026.08.01",
        "url": "https://example.com/1", "doc_type": "article",
    }], ensure_ascii=False), encoding="utf-8")


def draft_record():
    return {
        "query_id": "q1", "question": "질문", "construction_method": "query_first",
        "category": "entity", "seed_article_id": None, "gold_article_ids": [], "evidence": [],
        "author": "a", "reviewer": "", "review_status": "draft", "annotation_minutes": None,
        "naturalness": None, "gold_clarity": None, "notes": "",
    }


def test_validate_requires_allow_draft_for_draft_records(tmp_path):
    corpus, questions = tmp_path / "articles.json", tmp_path / "questions.jsonl"
    article_file(corpus)
    questions.write_text(json.dumps(draft_record(), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="draft"):
        main(["validate", str(questions), "--corpus", str(corpus)])
    assert main(["validate", str(questions), "--corpus", str(corpus), "--allow-draft"]) == 0


def test_empty_run_does_not_load_retrievers_or_write_results(tmp_path, capsys):
    corpus, questions = tmp_path / "articles.json", tmp_path / "questions.jsonl"
    article_file(corpus)
    questions.write_text("", encoding="utf-8")
    called = False
    def factory():
        nonlocal called
        called = True
        raise AssertionError("must not load")
    output = tmp_path / "result.json"
    assert main(["run", str(questions), "--corpus", str(corpus), "--json-output", str(output)], retriever_factory=factory) == 0
    assert not called and not output.exists()
    assert "no evaluation records" in capsys.readouterr().out.lower()


def test_pool_blind_hides_system_provenance(tmp_path, capsys):
    class Fake:
        def __init__(self, article_id): self.article_id = article_id
        def retrieve(self, question, top_k=5):
            return [{"chunk_id": str(self.article_id), "article_id": self.article_id, "title": "A", "date": "d", "content": "c", "url": "u", "doc_type": "article"}]
    factory = lambda: {"dense": Fake(1), "hybrid": Fake(2)}
    assert main(["pool", "질문", "--blind"], retriever_factory=factory) == 0
    output = json.loads(capsys.readouterr().out)
    assert "dense_rank" not in output[0]
    assert output[0]["candidate_id"] == "candidate_001"


def test_module_help_does_not_load_models():
    result = subprocess.run([sys.executable, "-m", "evaluation.cli", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "validate" in result.stdout and "pool" in result.stdout
