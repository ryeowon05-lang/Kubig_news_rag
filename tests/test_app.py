"""Smoke tests for the Streamlit UI.

No question is submitted, so no retriever is constructed and BGE-M3 never loads —
these stay fast. They catch the failure that matters most in a Streamlit script:
a layout or import error that only surfaces when the script body runs.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths against the calling file, not the project root.
APP = Path(__file__).resolve().parent.parent / "app.py"


def run_app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_renders_without_raising():
    app = run_app()

    assert not app.exception
    assert app.title[0].value.endswith("한국 금융 뉴스 RAG")


def test_both_retrieval_modes_are_offered_with_hybrid_first():
    app = run_app()

    assert app.sidebar.radio[0].options == ["Hybrid (Dense + BM25)", "Dense only"]
    assert app.sidebar.radio[0].value == "Hybrid (Dense + BM25)"


def test_retrieval_only_is_forced_when_no_api_key_is_configured(monkeypatch):
    # Answering costs API credits, so the UI must not offer generation it cannot do.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = run_app()

    checkbox = app.sidebar.checkbox[0]
    assert checkbox.value is True
    assert checkbox.disabled is True
    assert any("ANTHROPIC_API_KEY" in w.value for w in app.sidebar.warning)


def test_no_question_runs_no_retrieval():
    app = run_app()

    assert not app.subheader
    assert not app.expander
