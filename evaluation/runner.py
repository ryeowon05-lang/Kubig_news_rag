"""Run paired article-level Dense and Hybrid retrieval evaluation."""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics import dedupe_articles, score_ranking, top_k_overlap

DEFAULT_CHUNK_DEPTH = 100


def _mean(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _system_result(record: dict, system: str, retriever, chunk_depth: int) -> dict:
    started = time.perf_counter()
    error = None
    hits = []
    try:
        hits = retriever.retrieve(record["question"], top_k=chunk_depth)
    except Exception as exc:  # A paired evaluation keeps the other system result.
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started
    raw = [
        {"rank": rank, "chunk_id": hit.get("chunk_id"), "article_id": hit.get("article_id")}
        for rank, hit in enumerate(hits, 1)
    ]
    unique = dedupe_articles(hits, top_k=5) if not error else []
    scores = score_ranking([item["article_id"] for item in unique], set(record["gold_article_ids"]))
    return {
        "query_id": record["query_id"],
        "construction_method": record["construction_method"],
        "question": record["question"],
        "category": record["category"],
        "system": system,
        "raw_chunk_ranking": raw,
        "unique_article_ranking": unique,
        **scores,
        "top_5_overlap": 0,
        "latency_seconds": latency,
        "error": error,
    }


def _summaries(records: list[dict], results: list[dict]) -> list[dict]:
    summaries = []
    for method in ("source_seeded", "query_first"):
        selected = [record for record in records if record["construction_method"] == method]
        if not selected:
            continue
        ids = {record["query_id"] for record in selected}
        system_summary = {}
        for system in ("dense", "hybrid"):
            rows = [result for result in results if result["query_id"] in ids and result["system"] == system]
            system_summary[system] = {
                metric: _mean([row[metric] for row in rows])
                for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr_at_5")
            }
        summaries.append({
            "construction_method": method,
            "question_count": len(selected),
            "mean_annotation_minutes": _mean([record["annotation_minutes"] for record in selected]),
            "mean_naturalness": _mean([record["naturalness"] for record in selected]),
            "mean_gold_clarity": _mean([record["gold_clarity"] for record in selected]),
            "gold_article_count": sum(len(record["gold_article_ids"]) for record in selected),
            "mean_top_5_overlap": _mean([
                row["top_5_overlap"] for row in results if row["query_id"] in ids and row["system"] == "dense"
            ]),
            "uncertain_evidence_count": sum(
                evidence.get("relevance") == "uncertain"
                for record in selected for evidence in record["evidence"]
            ),
            "systems": system_summary,
            "warning": "Averages from 2–4 pilot questions are highly unstable and descriptive only.",
        })
    return summaries


def run_evaluation(
    records: list[dict],
    retrievers: dict[str, object],
    *,
    question_path: Path | str,
    corpus_article_count: int,
    corpus_chunk_count: int,
    settings: dict,
    git_commit: str,
    chunk_depth: int = DEFAULT_CHUNK_DEPTH,
    now: Callable[[], str] | None = None,
) -> dict:
    if set(retrievers) != {"dense", "hybrid"}:
        raise ValueError("retrievers must contain exactly dense and hybrid")
    path = Path(question_path)
    results = []
    for record in records:
        paired = {
            system: _system_result(record, system, retrievers[system], chunk_depth)
            for system in ("dense", "hybrid")
        }
        overlap = top_k_overlap(
            [item["article_id"] for item in paired["dense"]["unique_article_ranking"]],
            [item["article_id"] for item in paired["hybrid"]["unique_article_ranking"]],
        )
        paired["dense"]["top_5_overlap"] = overlap
        paired["hybrid"]["top_5_overlap"] = overlap
        results.extend((paired["dense"], paired["hybrid"]))
    timestamp = now() if now else datetime.now(timezone.utc).isoformat()
    return {
        "metadata": {
            "executed_at": timestamp,
            "git_commit": git_commit,
            "question_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "corpus": {"article_count": corpus_article_count, "chunk_count": corpus_chunk_count},
            "retrieval_settings": {**settings, "evaluation_chunk_depth": chunk_depth, "article_top_k": 5},
        },
        "results": results,
        "summaries": _summaries(records, results),
    }
