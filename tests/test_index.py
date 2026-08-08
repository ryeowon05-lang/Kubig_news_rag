import json
import subprocess
import sys

from indexing.build_index import build_index, deserialize_metadata, serialize_metadata
from indexing.search import search_collection


class DeterministicEmbedder:
    def encode(self, texts, batch_size=None):
        vectors = []
        for text in texts:
            vectors.append([1.0, 0.0] if "반도체" in text else [0.0, 1.0])
        return vectors


def chunk(chunk_id, text, names):
    return {
        "chunk_id": chunk_id,
        "article_id": 1,
        "chunk_index": 0,
        "title": "제목",
        "content": text,
        "embedding_text": text,
        "date": "2026.08.08",
        "url": f"https://example.com/{chunk_id}",
        "stock_names": names,
        "stock_codes": ["005930", "000660"][: len(names)],
        "source_ids": [1, 2][: len(names)],
        "doc_type": "article",
    }


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def test_metadata_lists_round_trip_as_json():
    original = chunk("a", "반도체", ["삼성전자", "SK하이닉스"])
    stored = serialize_metadata(original)
    assert stored["stock_names"] == '["삼성전자", "SK하이닉스"]'
    assert stored["source_ids"] == "[1, 2]"
    restored = deserialize_metadata(stored)
    assert restored["stock_names"] == ["삼성전자", "SK하이닉스"]
    assert restored["stock_codes"] == ["005930", "000660"]
    assert restored["source_ids"] == [1, 2]


def test_upsert_is_idempotent_and_cosine_query_works(tmp_path):
    source = tmp_path / "chunks.jsonl"
    records = [chunk("semiconductor", "반도체 실적", ["삼성전자"]), chunk("battery", "배터리 수요", ["LG에너지솔루션"])]
    write_jsonl(source, records)
    embedder = DeterministicEmbedder()

    collection = build_index(source, tmp_path / "chroma", "test_news", embedder, batch_size=1)
    build_index(source, tmp_path / "chroma", "test_news", embedder, batch_size=2)

    assert collection.count() == 2
    result = collection.query(query_embeddings=[[1.0, 0.0]], n_results=2, include=["metadatas", "distances"])
    assert result["ids"][0][0] == "semiconductor"
    assert result["distances"][0][0] == 0.0
    assert deserialize_metadata(result["metadatas"][0][0])["stock_names"] == ["삼성전자"]


def test_search_restores_list_metadata(tmp_path):
    source = tmp_path / "chunks.jsonl"
    write_jsonl(source, [chunk("semiconductor", "반도체 실적", ["삼성전자", "SK하이닉스"])])
    embedder = DeterministicEmbedder()
    collection = build_index(source, tmp_path / "chroma", "test_search", embedder)

    results = search_collection(collection, embedder, "반도체 전망", top_k=5)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "semiconductor"
    assert results[0]["content"] == "반도체 실적"
    assert results[0]["stock_names"] == ["삼성전자", "SK하이닉스"]
    assert results[0]["distance"] == 0.0


def test_search_script_can_run_directly():
    result = subprocess.run(
        [sys.executable, "indexing/search.py", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
