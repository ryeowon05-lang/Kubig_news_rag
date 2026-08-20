# Retrieval Evaluation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a human-authored, article-level pilot framework that validates, pools, runs, and reports paired Dense and Hybrid retrieval evaluation.

**Architecture:** A standalone `evaluation` package keeps annotation and metric policy out of production retrieval code. Pure schema/metric functions feed an injected-retriever runner; a thin CLI connects existing loaders and writes deterministic JSON/CSV artifacts.

**Tech Stack:** Python 3.13 standard library, existing DenseRetriever/HybridRetriever, pytest

**Spec:** `docs/superpowers/specs/2026-08-20-retrieval-eval-pilot-design.md`

## Global Constraints

- Do not modify retrieval settings, source data, cleaned data, chunks, Chroma, or RRF behavior.
- Do not generate or approve evaluation questions.
- Use article-level Hit@1/3/5 and MRR@5 as primary metrics; chunk rankings are diagnostic.
- Do not add dependencies or expose secrets, environment values, or local paths.

---

### Task 1: Schema validation and article browser

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/schema.py`
- Create: `evaluation/articles.py`
- Create: `evaluation/pilot_retrieval_eval.jsonl`
- Create: `evaluation/pilot_retrieval_eval.example.jsonl`
- Test: `tests/test_evaluation_schema.py`
- Test: `tests/test_evaluation_articles.py`

**Interfaces:**
- Produces: `load_records(path) -> list[dict]`, `validate_records(records, article_ids, *, allow_draft) -> None`, `ArticleCatalog.from_json(path)`, and lookup/search/sample methods.

- [ ] Write failing table-driven tests for every schema invariant, final/draft policy, lookup, keyword search, deterministic sampling, and absence of scores.
- [ ] Run `uv run pytest -q tests/test_evaluation_schema.py tests/test_evaluation_articles.py` and confirm missing-module failure.
- [ ] Implement strict JSONL loading, validation, and read-only corpus browsing with `random.Random(seed)`.
- [ ] Add an empty real JSONL and one unmistakable draft placeholder to the explicit example file.
- [ ] Re-run the focused tests and confirm PASS.

### Task 2: Article ranking, pooling, and metrics

**Files:**
- Create: `evaluation/metrics.py`
- Create: `evaluation/pooling.py`
- Test: `tests/test_evaluation_metrics.py`
- Test: `tests/test_evaluation_pooling.py`

**Interfaces:**
- Consumes: retriever hits containing `chunk_id` and `article_id`.
- Produces: `dedupe_articles(hits, top_k)`, `score_ranking(article_ids, gold_ids)`, and `build_pool(dense_hits, hybrid_hits, depth, blind)`.

- [ ] Write failing tests for duplicate chunks, first-rank preservation, unique top-five after over-retrieval, multiple golds, misses, Hit@1/3/5, MRR@5, overlap, pooled deduplication, and blind output.
- [ ] Run focused tests and confirm missing behavior failures.
- [ ] Implement pure ranking and pooling functions with literal, stable result dictionaries.
- [ ] Re-run focused tests and confirm PASS.

### Task 3: Paired runner and reproducible reporting

**Files:**
- Create: `evaluation/runner.py`
- Create: `evaluation/reporting.py`
- Test: `tests/test_evaluation_runner.py`
- Test: `tests/test_evaluation_reporting.py`

**Interfaces:**
- Consumes: validated records, injected `dict[str, Retriever]`, corpus/chunk counts, retrieval settings, explicit question path.
- Produces: paired per-system results, construction-method summaries, `write_json`, and `write_csv`.

- [ ] Write failing fake-retriever tests for paired calls at chunk depth 100, raw/unique rankings, latency/error, SHA-256, Git/settings metadata, summaries, uncertain counts, and flat CSV rows.
- [ ] Run focused tests and confirm missing behavior failures.
- [ ] Implement per-system exception capture, deterministic metadata serialization, aggregation, and output writers.
- [ ] Ensure empty input returns no results and the CLI can skip artifact creation.
- [ ] Re-run focused tests and confirm PASS.

### Task 4: CLI and human workflow documentation

**Files:**
- Create: `evaluation/cli.py`
- Create: `evaluation/README.md`
- Test: `tests/test_evaluation_cli.py`

**Interfaces:**
- Consumes: explicit CLI paths and existing retriever factories.
- Produces: `articles get/search/sample`, `pool`, `validate`, and `run` commands.

- [ ] Write failing CLI tests for explicit input paths, `--allow-draft`, blind pools, empty-run behavior, and help output without loading models.
- [ ] Run focused tests and confirm failures.
- [ ] Implement argparse commands and lazy production retriever construction.
- [ ] Document both construction methods, 2-person workflow, full-corpus rule, metrics, limitations, commands, and adding questions.
- [ ] Re-run CLI tests and confirm PASS.

### Task 5: Verification and publication

**Files:** all evaluation framework files listed above.

**Interfaces:** no new interfaces.

- [ ] Run `uv run pytest -q` and confirm zero failures.
- [ ] Run `uv lock --check` and confirm the lock is unchanged and valid.
- [ ] Run `git diff --check` and inspect `git diff` and `git status --short`.
- [ ] Stage only evaluation framework, tests, README/spec/plan files explicitly.
- [ ] Run `git diff --cached --check` and inspect `git diff --cached`.
- [ ] Commit with `feat: add retrieval evaluation pilot framework`.
- [ ] Push `feat/retrieval-eval-pilot` without merging to `main`.
