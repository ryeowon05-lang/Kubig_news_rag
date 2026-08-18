"""Claude API adapter used by the RAG generation stage."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 1024


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce an answer."""


class ClaudeClient:
    """Small adapter around Anthropic's Messages API.

    The Anthropic client is imported lazily so prompt/retrieval unit tests do not
    require an API key or an active network connection.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        self.model = model or os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens

        if client is not None:
            self._client = client
            return

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Put your Claude API key in the environment."
            )

        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required. Run: uv add anthropic"
            ) from exc

        self._client = Anthropic(api_key=key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one text answer from Claude."""
        if not system_prompt.strip() or not user_prompt.strip():
            raise ValueError("system_prompt and user_prompt cannot be empty")

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as exc:
            raise LLMError(f"Claude API request failed: {exc}") from exc

        text_parts = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        ]
        answer = "\n".join(text_parts).strip()
        if not answer:
            raise LLMError("Claude returned an empty text response")
        return answer
