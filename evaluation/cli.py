"""CLI for human-authored Dense versus Hybrid retrieval evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from evaluation.articles import ArticleCatalog
from evaluation.pooling import build_pool
from evaluation.reporting import write_csv, write_json
from evaluation.runner import DEFAULT_CHUNK_DEPTH, run_evaluation
from evaluation.schema import load_records, validate_records
from indexing.build_index import DEFAULT_COLLECTION, DEFAULT_INPUT, DEFAULT_MODEL
from indexing.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "processed" / "news_data_clean.json"
DEFAULT_RESULT = PROJECT_ROOT / "evaluation" / "results" / "retrieval_eval.json"


def _retrievers() -> dict:
    from indexing.hybrid import HybridRetriever
    from indexing.retriever import detect_device, get_retriever

    device = detect_device()
    dense = get_retriever(device=device)
    return {"dense": dense, "hybrid": HybridRetriever(dense=dense, device=device)}


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    articles = subparsers.add_parser("articles", help="browse cleaned articles without retrieval scores")
    article_commands = articles.add_subparsers(dest="article_command", required=True)
    get = article_commands.add_parser("get")
    get.add_argument("article_id", type=int)
    search = article_commands.add_parser("search")
    search.add_argument("keyword")
    search.add_argument("--limit", type=int, default=20)
    sample = article_commands.add_parser("sample")
    sample.add_argument("--count", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)
    for command in (get, search, sample):
        command.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    validate = subparsers.add_parser("validate", help="validate an explicitly named evaluation JSONL")
    validate.add_argument("evaluation_file", type=Path)
    validate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    validate.add_argument("--allow-draft", action="store_true")

    pool = subparsers.add_parser("pool", help="pool Dense and Hybrid candidates for query-first labeling")
    pool.add_argument("question")
    pool.add_argument("--blind", action="store_true")
    pool.add_argument("--pool-depth", type=int, default=20)
    pool.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH)

    run = subparsers.add_parser("run", help="run paired article-level evaluation")
    run.add_argument("evaluation_file", type=Path)
    run.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    run.add_argument("--chunks", type=Path, default=DEFAULT_INPUT)
    run.add_argument("--allow-draft", action="store_true")
    run.add_argument("--chunk-depth", type=int, default=DEFAULT_CHUNK_DEPTH)
    run.add_argument("--json-output", type=Path, default=DEFAULT_RESULT)
    run.add_argument("--csv-output", type=Path)
    return parser


def main(argv: list[str] | None = None, *, retriever_factory=None) -> int:
    args = build_parser().parse_args(argv)
    factory = retriever_factory or _retrievers
    if args.command == "articles":
        catalog = ArticleCatalog.from_json(args.corpus)
        if args.article_command == "get":
            value = catalog.get(args.article_id)
        elif args.article_command == "search":
            if args.limit < 1:
                raise ValueError("limit must be at least 1")
            value = catalog.search(args.keyword)[: args.limit]
        else:
            value = catalog.sample(args.count, seed=args.seed)
        _print(value)
        return 0

    if args.command == "pool":
        retrievers = factory()
        hits = {
            system: retriever.retrieve(args.question, top_k=args.chunk_depth)
            for system, retriever in retrievers.items()
        }
        _print(build_pool(hits["dense"], hits["hybrid"], depth=args.pool_depth, blind=args.blind))
        return 0

    catalog = ArticleCatalog.from_json(args.corpus)
    records = load_records(args.evaluation_file)
    validate_records(records, catalog.article_ids, allow_draft=args.allow_draft)
    if args.command == "validate":
        print(f"valid: {len(records)} records")
        return 0
    if not records:
        print("No evaluation records; no result artifacts were written.")
        return 0

    if not args.chunks.is_file():
        raise FileNotFoundError(f"chunk file not found: {args.chunks}")
    report = run_evaluation(
        records, factory(), question_path=args.evaluation_file,
        corpus_article_count=len(catalog.article_ids), corpus_chunk_count=_count_jsonl(args.chunks),
        settings={
            "systems": ["dense", "hybrid"], "model": DEFAULT_MODEL,
            "collection": DEFAULT_COLLECTION, "candidate_k": DEFAULT_CANDIDATE_K, "rrf_k": DEFAULT_RRF_K,
        },
        git_commit=_git_commit(), chunk_depth=args.chunk_depth,
    )
    json_output = args.json_output
    csv_output = args.csv_output or json_output.with_suffix(".csv")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, json_output)
    write_csv(report, csv_output)
    print(f"wrote {json_output} and {csv_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
