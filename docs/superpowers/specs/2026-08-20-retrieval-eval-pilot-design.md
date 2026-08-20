# Retrieval Evaluation Pilot Design

## Purpose

Build a human-authored pilot framework for comparing the existing Dense and
Hybrid retrievers at article level. The framework supports eight eventual
questions across `source_seeded` and `query_first`, but it does not generate or
approve questions and does not produce performance results while the real
evaluation file is empty.

## Boundaries

- Search always uses the full cleaned corpus and existing full chunk index.
- Existing model, Chroma, BM25, `candidate_k=20`, `rrf_k=60`, and RRF behavior
  remain unchanged.
- Source data, cleaned data, chunks, and indexes remain read-only.
- Primary metrics are article-level; raw chunk rankings are diagnostic only.
- No model superiority, significance, or question-construction recommendation
  is inferred from this small pilot.

## Components

- `evaluation/schema.py`: load JSONL and validate record, review, evidence, and
  corpus-reference invariants.
- `evaluation/articles.py`: read-only article lookup, keyword search, and seeded
  random sampling without retrieval scores.
- `evaluation/pooling.py`: pool Dense and Hybrid candidates by `article_id` and
  optionally hide system names and ranks during relevance judgment.
- `evaluation/metrics.py`: preserve raw chunk order, deduplicate by first article
  occurrence, and calculate Hit@1/3/5, MRR@5, first relevant rank, and overlap.
- `evaluation/runner.py`: paired execution with injected retrievers, timing,
  per-question error capture, corpus counts, Git commit, settings, and SHA-256.
- `evaluation/reporting.py`: deterministic JSON and flat CSV output plus small-
  sample construction-method summaries.
- `evaluation/cli.py`: explicit `articles`, `pool`, `validate`, and `run`
  subcommands. Evaluation paths are explicit; example data is never discovered
  automatically.

## Data Flow

1. Load the explicitly named evaluation JSONL and cleaned article corpus.
2. Validate all records and referenced article IDs before retrieval.
3. For each question, ask both retrievers for an evaluation-only chunk depth
   sufficient to form five unique articles. This does not alter retriever
   internals or tune Hybrid settings.
4. Save raw chunk IDs and article IDs, then deduplicate articles while preserving
   the first occurrence.
5. Calculate paired article-level metrics and top-five overlap.
6. Write one reproducible JSON artifact and one flat CSV artifact, excluding
   secrets, environment values, local paths, and model cache paths.
7. Summarize each construction method descriptively with an instability warning.

## Validation and Failure Policy

- Schema, duplicate IDs, invalid categories, missing seed IDs, invalid corpus
  references, invalid gold/evidence relationships, and review conflicts fail
  before retrieval.
- Final runs reject drafts and uncertain evidence. Pilot runs allow drafts only
  with an explicit `--allow-draft` flag.
- An individual retriever failure is recorded in that system result's `error`
  field so its paired result remains inspectable; file and configuration errors
  fail the command.
- An empty real evaluation file validates successfully but `run` exits without
  writing misleading performance artifacts.

## Defaults

- Article evaluation depth: top five unique articles.
- Pool depth: top twenty unique articles per system.
- Chunk over-retrieval depth: 100, documented as an evaluation implementation
  detail rather than a change to `candidate_k`.
- Random article sampling seed: 42.
- Allowed categories: `exact_token`, `entity`, `semantic_paraphrase`,
  `multi_entity`, and `temporal_numeric`.

## Testing

Tests use fake retrievers and temporary files only. They cover article
deduplication/order, over-retrieval, multiple golds, metrics, schema/review
rules, paired outputs, aggregation, raw and unique rankings, and question-file
SHA-256. Existing tests must continue to pass without loading BGE-M3, opening a
real Chroma index, or calling Claude.
