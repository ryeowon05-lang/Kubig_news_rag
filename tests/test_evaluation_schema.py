import json

import pytest

from evaluation.schema import load_records, validate_records


def record(**changes):
    value = {
        "query_id": "q1",
        "question": "반도체 투자 전망은?",
        "construction_method": "source_seeded",
        "category": "semantic_paraphrase",
        "seed_article_id": 1,
        "gold_article_ids": [1],
        "evidence": [{"article_id": 1, "title": "기사", "support": "근거", "relevance": "relevant"}],
        "author": "annotator_a",
        "reviewer": "annotator_b",
        "review_status": "approved",
        "annotation_minutes": 8.5,
        "naturalness": 5,
        "gold_clarity": 4,
        "notes": "",
    }
    value.update(changes)
    return value


def test_load_records_reads_nonblank_jsonl_lines(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("\n" + json.dumps(record(), ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_records(path)[0]["query_id"] == "q1"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"question": " "}, "question"),
        ({"construction_method": "generated"}, "construction_method"),
        ({"category": ""}, "category"),
        ({"seed_article_id": None}, "seed_article_id"),
        ({"gold_article_ids": [1, 1]}, "duplicate"),
        ({"gold_article_ids": [999]}, "corpus"),
        ({"evidence": [{"article_id": 2, "title": "x", "support": "x", "relevance": "relevant"}]}, "gold_article_ids"),
        ({"author": "same", "reviewer": "same"}, "author"),
    ],
)
def test_validate_records_rejects_invalid_records(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_records([record(**changes)], {1, 2}, allow_draft=False)


def test_query_first_allows_null_seed_article_id():
    validate_records(
        [record(construction_method="query_first", seed_article_id=None)],
        {1},
        allow_draft=False,
    )


def test_duplicate_query_ids_are_rejected():
    with pytest.raises(ValueError, match="query_id"):
        validate_records([record(), record()], {1}, allow_draft=False)


def test_draft_requires_explicit_allow_draft():
    draft = record(review_status="draft", gold_article_ids=[], evidence=[], reviewer="")
    with pytest.raises(ValueError, match="draft"):
        validate_records([draft], {1}, allow_draft=False)
    validate_records([draft], {1}, allow_draft=True)


def test_final_rejects_uncertain_evidence():
    uncertain = record(evidence=[{"article_id": 1, "title": "기사", "support": "검토 중", "relevance": "uncertain"}])
    with pytest.raises(ValueError, match="uncertain"):
        validate_records([uncertain], {1}, allow_draft=False)


def test_approved_requires_at_least_one_gold_article():
    with pytest.raises(ValueError, match="gold_article_ids"):
        validate_records([record(gold_article_ids=[], evidence=[])], {1}, allow_draft=False)


def test_evidence_requires_article_title_support_and_relevance_fields():
    with pytest.raises(ValueError, match="evidence"):
        validate_records([record(evidence=[{"article_id": 1, "relevance": "relevant"}])], {1}, allow_draft=False)
