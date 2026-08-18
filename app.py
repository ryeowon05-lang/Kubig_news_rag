"""Streamlit UI for the Korean financial-news RAG pipeline.

Two retrieval modes (dense-only and hybrid) over the same index, so the two can
be compared on the same question. Retrieval-only mode skips the Claude call —
use it while tuning retrieval, since every answer costs API credits.
"""

from __future__ import annotations

import os

import streamlit as st

from indexing.build_index import DEFAULT_DB
from rag import RAGPipeline

st.set_page_config(page_title="한국 금융 뉴스 RAG", page_icon="📰", layout="wide")


def detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner="BGE-M3 로딩 중… (최초 1회, 10초쯤 걸립니다)")
def load_dense(device: str):
    from indexing.retriever import get_retriever

    retriever = get_retriever(device=device)
    retriever.retrieve("워밍업")  # 모델·컬렉션을 첫 질문 전에 올려둔다
    return retriever


@st.cache_resource(show_spinner="BM25 인덱스 구축 중… (최초 1회, 5초쯤 걸립니다)")
def load_bm25():
    from indexing.bm25 import get_bm25_retriever

    retriever = get_bm25_retriever()
    retriever.retrieve("워밍업")
    return retriever


@st.cache_resource
def load_hybrid(device: str, candidate_k: int, rrf_k: int):
    from indexing.hybrid import HybridRetriever

    return HybridRetriever(
        dense=load_dense(device),
        bm25=load_bm25(),
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )


@st.cache_resource
def load_llm():
    from generation.llm import ClaudeClient

    return ClaudeClient()


def render_chunk(rank: int, chunk: dict) -> None:
    """One retrieved chunk, with where it came from."""
    title = chunk.get("title") or "(제목 없음)"
    with st.expander(f"**{rank}. {title}**", expanded=rank <= 2):
        bits = []
        if chunk.get("date"):
            bits.append(chunk["date"])
        if chunk.get("stock_names"):
            bits.append("· ".join(chunk["stock_names"]))
        if chunk.get("doc_type") == "broadcast":
            bits.append("방송")
        st.caption(" | ".join(bits))

        provenance = []
        if chunk.get("rrf_score") is not None:
            dense_rank = chunk.get("dense_rank")
            bm25_rank = chunk.get("bm25_rank")
            provenance.append(f"RRF {chunk['rrf_score']:.4f}")
            provenance.append(f"dense #{dense_rank}" if dense_rank else "dense —")
            provenance.append(f"bm25 #{bm25_rank}" if bm25_rank else "bm25 —")
        if chunk.get("similarity") is not None:
            provenance.append(f"유사도 {chunk['similarity']:.4f}")
        if chunk.get("bm25_score") is not None:
            provenance.append(f"BM25 {chunk['bm25_score']:.2f}")
        if chunk.get("cited_as"):
            provenance.append("인용 " + ", ".join(f"[뉴스{n}]" for n in chunk["cited_as"]))
        if provenance:
            st.caption(" · ".join(provenance))

        st.write(chunk.get("content", ""))
        if chunk.get("url"):
            st.markdown(f"[원문 보기]({chunk['url']})")


# --- page -----------------------------------------------------------------

# Rendered before the sidebar so the page still has a heading when a later check
# stops the script — a bare st.stop() with nothing drawn yet is a blank page.
st.title("📰 한국 금융 뉴스 RAG")

with st.sidebar:
    st.header("검색 설정")
    mode = st.radio("검색 방식", ["Hybrid (Dense + BM25)", "Dense only"], index=0)
    is_hybrid = mode.startswith("Hybrid")

    top_k = st.slider("top_k (최종 청크 수)", 1, 10, 5)
    candidate_k = st.slider("후보 풀 (각 검색기당)", 5, 50, 20, disabled=not is_hybrid)
    rrf_k = st.slider("RRF k", 10, 120, 60, disabled=not is_hybrid)

    st.divider()
    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    retrieval_only = st.checkbox(
        "검색만 (LLM 호출 안 함)",
        value=not has_key,
        disabled=not has_key,
        help="검색 결과만 봅니다. 답변 생성에는 Claude API 비용이 듭니다.",
    )
    if not has_key:
        st.warning("ANTHROPIC_API_KEY가 없어 검색 전용으로 동작합니다.")

    device = st.selectbox(
        "임베딩 device", ["auto", "cpu", "cuda", "mps"], index=0
    )
    resolved_device = detect_device() if device == "auto" else device
    st.caption(f"device: `{resolved_device}`")

st.caption(
    "네이버 금융 뉴스 432건 · 1,377청크 · BGE-M3 임베딩 + BM25 · "
    f"현재 모드: **{mode}**"
)

if not DEFAULT_DB.is_dir():
    st.error(
        f"Chroma 인덱스가 없습니다 (`{DEFAULT_DB}`). 먼저 인덱스를 만들어 주세요.\n\n"
        "```\n"
        "uv run python indexing/chunk.py\n"
        "uv run python indexing/build_index.py --rebuild --device mps\n"
        "```"
    )
    st.stop()

question = st.text_input(
    "질문", placeholder="예: 삼전 반도체 실적 전망은?", label_visibility="collapsed"
)

if question:
    retriever = (
        load_hybrid(resolved_device, candidate_k, rrf_k)
        if is_hybrid
        else load_dense(resolved_device)
    )

    if retrieval_only:
        with st.spinner("검색 중…"):
            chunks = retriever.retrieve(question, top_k=top_k)
        st.subheader(f"검색 결과 {len(chunks)}개")
        if not chunks:
            st.info("검색 결과가 없습니다.")
        for rank, chunk in enumerate(chunks, 1):
            render_chunk(rank, chunk)
    else:
        with st.spinner("검색하고 답변 생성 중…"):
            response = RAGPipeline(retriever=retriever, llm=load_llm()).ask(
                question, top_k=top_k
            )
        st.subheader("답변")
        st.write(response.answer)
        st.subheader(f"출처 {len(response.sources)}건")
        if not response.sources:
            st.info("관련 뉴스를 찾지 못했습니다.")
        for source in response.sources:
            render_chunk(source["news_number"], source)
