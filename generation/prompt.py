"""Prompt construction for grounded Korean financial-news QA."""

from __future__ import annotations

SYSTEM_PROMPT = """당신은 한국 금융 뉴스 기반 질의응답 시스템의 답변 생성기입니다.

반드시 제공된 뉴스 자료만 근거로 답변하세요.
- 제공된 자료에 없는 사실, 수치, 원인, 전망을 추측하거나 외부 지식을 사용하지 마세요.
- 자료만으로 질문에 답할 수 없다면 정확히 "제공된 자료에서 확인할 수 없습니다."라고 답하세요.
- 질문이 요구하는 범위에 맞게 간결하고 명확한 한국어로 답변하세요.
- 답변에 근거가 된 뉴스 번호를 [뉴스1], [뉴스2]와 같은 형식으로 표시하세요.
- 서로 다른 뉴스의 내용이 다르면 임의로 하나를 선택하지 말고 차이를 밝혀 주세요.
- 투자 판단을 직접 지시하지 말고, 뉴스에서 확인되는 사실과 전망을 구분해서 설명하세요.
"""


def _format_stocks(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def build_user_prompt(question: str, results: list[dict]) -> str:
    """Build the grounded user prompt from retrieval results."""
    if not question.strip():
        raise ValueError("question cannot be empty")
    if not results:
        raise ValueError("results cannot be empty")

    contexts: list[str] = []
    for rank, result in enumerate(results, 1):
        contexts.append(
            f"""[뉴스{rank}]
제목: {result.get('title', '')}
날짜: {result.get('date', '')}
종목: {_format_stocks(result.get('stock_names', []))}
URL: {result.get('url', '')}
본문:
{result.get('content', '')}
""".strip()
        )

    return """다음 뉴스 자료를 근거로 질문에 답하세요.

[뉴스 자료]

{contexts}

[질문]
{question}

[답변 작성 규칙]
1. 뉴스 자료에 직접 근거가 있는 내용만 답하세요.
2. 답변의 관련 문장 끝에 근거 뉴스 번호를 표시하세요. 예: [뉴스1]
3. 근거를 찾을 수 없으면 "제공된 자료에서 확인할 수 없습니다."라고 답하세요.
4. 답변에서 URL을 직접 나열할 필요는 없습니다. 출처 정보는 시스템이 별도로 제공합니다.
""".format(contexts="\n\n".join(contexts), question=question.strip())


def build_messages(question: str, results: list[dict]) -> tuple[str, str]:
    """Return the system prompt and user prompt for the Claude Messages API."""
    return SYSTEM_PROMPT, build_user_prompt(question, results)
