"""Ask the Korean financial-news index a question from the terminal.

The same pipeline the Streamlit app drives, with every retrieval technique
reachable from a flag instead of a widget:

    uv run python cli.py "삼성전자 반도체 실적 전망은?"
    uv run python cli.py "005930 주가" --mode bm25
    uv run python cli.py "HBM 수요" --mode hybrid --expand hyde
    uv run python cli.py "코스피 시황" --retrieval-only      # API 크레딧 안 씀
    uv run python cli.py --chat --gate                      # 후속 질문 가능

`--retrieval-only` needs no API key at all, which makes it the cheap way to tune
retrieval; everything else streams the answer as Claude writes it.
"""

from __future__ import annotations

import argparse
import json
import sys

from indexing.hybrid import DEFAULT_CANDIDATE_K, DEFAULT_RRF_K
from rag import DEFAULT_TOP_K, RAGPipeline
from ui_format import meta_line, provenance_line

MODES = ("dense", "bm25", "hybrid")
EXPANDERS = ("none", "multi_query", "hyde")
EXIT_WORDS = {"exit", "quit", "q", "종료", "나가기"}
SNIPPET_CHARS = 160
RULE = "─" * 60


class Loaders:
    """Builds the heavy pieces once, on first use.

    Injected rather than imported at the point of use so the CLI can be wired
    and tested without loading a 4.5GB embedding model or holding an API key.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._dense = None
        self._bm25 = None
        self._llm = None

    def dense(self):
        if self._dense is None:
            from indexing.retriever import get_retriever

            self._dense = get_retriever(device=self.device)
        return self._dense

    def bm25(self):
        if self._bm25 is None:
            from indexing.bm25 import get_bm25_retriever

            self._bm25 = get_bm25_retriever()
        return self._bm25

    def llm(self):
        if self._llm is None:
            from generation.llm import ClaudeClient

            self._llm = ClaudeClient()
        return self._llm


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="한국 금융 뉴스 RAG — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="?", help="질문 (--chat 이면 생략)")
    parser.add_argument("--chat", action="store_true", help="후속 질문이 가능한 대화 모드")
    parser.add_argument("--mode", choices=MODES, default="hybrid", help="검색 방식")
    parser.add_argument("--expand", choices=EXPANDERS, default="none", help="질의 확장")
    parser.add_argument("--gate", action="store_true", help="LLM이 검색 필요 여부를 판단")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--retrieval-only", action="store_true", help="검색만 하고 Claude는 호출하지 않음"
    )
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")

    args = parser.parse_args(argv)
    if not args.chat and not args.question:
        parser.error("question is required unless --chat is given")
    if args.chat and args.retrieval_only:
        # 대화 모드의 핵심은 직전 답변을 기억하는 것인데, 답변을 만들지 않으면
        # 기억할 것도 없다. 검색만 볼 거라면 단발 질문으로 충분하다.
        parser.error("--retrieval-only is a single-question mode; drop --chat")
    return args


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    from indexing.retriever import detect_device

    return detect_device()


def build_retriever(args: argparse.Namespace, loaders: Loaders):
    """Assemble the retrieval stack the flags describe."""
    if args.mode == "dense":
        base = loaders.dense()
    elif args.mode == "bm25":
        base = loaders.bm25()
    else:
        from indexing.hybrid import HybridRetriever

        base = HybridRetriever(
            dense=loaders.dense(),
            bm25=loaders.bm25(),
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )

    if args.expand == "none":
        # 확장을 안 쓰면 LLM을 만들지 않는다 — --retrieval-only가 API 키 없이 도는 이유.
        return base

    from indexing.expanding import ExpandingRetriever
    from indexing.query_expand import get_expander

    return ExpandingRetriever(
        base=base,
        expander=get_expander(args.expand, llm=loaders.llm()),
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )


def build_pipeline(args: argparse.Namespace, loaders: Loaders) -> RAGPipeline:
    gate = None
    if args.gate:
        from generation.router import SearchGate

        gate = SearchGate(llm=loaders.llm())
    return RAGPipeline(
        retriever=build_retriever(args, loaders), llm=loaders.llm(), gate=gate
    )


def format_source(rank: int, source: dict) -> str:
    """One retrieved chunk as an indented block: title, meta, why, link, snippet."""
    lines = [f"  [{rank}] {source.get('title') or '(제목 없음)'}"]
    for line in (meta_line(source), provenance_line(source), source.get("url") or ""):
        if line:
            lines.append(f"      {line}")
    content = " ".join((source.get("content") or "").split())
    if content:
        lines.append(f"      {content[:SNIPPET_CHARS]}…")
    return "\n".join(lines)


def write_sources(sources: list[dict], out) -> None:
    if not sources:
        out.write("출처 없음 (검색을 건너뛰었거나 결과가 없습니다)\n")
        return
    out.write(f"출처 {len(sources)}건\n")
    for rank, source in enumerate(sources, 1):
        out.write(format_source(rank, source) + "\n")


def run_once(
    pipeline,
    question: str,
    args: argparse.Namespace,
    out=None,
    history: list[dict] | None = None,
) -> str:
    """Ask one question. Returns the answer so a chat turn can remember it."""
    out = out or sys.stdout
    response = pipeline.ask_stream(question, top_k=args.top_k, history=history)

    if args.json:
        answer = "".join(response.stream)
        json.dump(
            {
                "question": response.question,
                "answer": answer,
                "searched": response.searched,
                "sources": response.sources,
            },
            out,
            ensure_ascii=False,
            indent=2,
        )
        out.write("\n")
        return answer

    # 출처는 검색이 끝난 시점에 이미 확정돼 있다. 본문을 기다리지 않고 먼저 보여준다.
    write_sources(response.sources, out)
    out.write(RULE + "\n")
    pieces = []
    for piece in response.stream:
        out.write(piece)
        out.flush()
        pieces.append(piece)
    out.write("\n")
    return "".join(pieces)


def run_chat(pipeline, args: argparse.Namespace, out=None, input_fn=input) -> None:
    """Multi-turn loop. Each turn sees the previous ones, so follow-ups work."""
    out = out or sys.stdout
    history: list[dict] = []
    while True:
        try:
            question = input_fn("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            out.write("\n")
            return
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            return
        answer = run_once(pipeline, question, args, out=out, history=history)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def run_retrieval_only(args: argparse.Namespace, loaders: Loaders, out) -> None:
    results = build_retriever(args, loaders).retrieve(args.question, top_k=args.top_k)
    if args.json:
        json.dump(
            {"question": args.question, "answer": None, "sources": results},
            out,
            ensure_ascii=False,
            indent=2,
        )
        out.write("\n")
        return
    write_sources(results, out)


def main(argv: list[str] | None = None, loaders: Loaders | None = None, out=None) -> int:
    args = parse_args(argv)
    out = out or sys.stdout
    if loaders is None:
        loaders = Loaders(device=resolve_device(args.device))

    try:
        if args.retrieval_only:
            run_retrieval_only(args, loaders, out)
        elif args.chat:
            run_chat(build_pipeline(args, loaders), args, out=out)
        else:
            run_once(build_pipeline(args, loaders), args.question, args, out=out)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        # 인덱스 없음 / API 키 없음 / 컬렉션 불일치 — 스택 트레이스 대신 한 줄로.
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
