from evaluation.pooling import build_pool


def hit(chunk_id, article_id, title):
    return {
        "chunk_id": chunk_id, "article_id": article_id, "title": title, "date": "2026.08.01",
        "content": "본문", "url": "https://example.com", "doc_type": "article",
    }


def test_pool_deduplicates_by_article_and_records_both_system_ranks():
    pool = build_pool(
        [hit("a1", 1, "A"), hit("b", 2, "B")],
        [hit("a2", 1, "A"), hit("c", 3, "C")],
        depth=20,
        blind=False,
    )
    assert [item["article_id"] for item in pool] == [1, 2, 3]
    assert pool[0]["dense_rank"] == 1
    assert pool[0]["hybrid_rank"] == 1


def test_blind_pool_hides_system_names_and_ranks_but_keeps_article_content():
    pool = build_pool([hit("a", 1, "A")], [], depth=20, blind=True)
    assert set(pool[0]) == {"candidate_id", "article_id", "title", "date", "content", "url", "doc_type"}
    assert pool[0]["candidate_id"] == "candidate_001"
