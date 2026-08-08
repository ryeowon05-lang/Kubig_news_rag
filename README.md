# Kr News Rag

한국 금융 뉴스 기반 RAG 에이전트 프로젝트.

현재는 1주차 산출물인 네이버 금융 종목 뉴스 크롤러(`crawler.ipynb`)와
수집 결과(`news_data.json`, 10개 종목 × 30건 = 300건)가 포함되어 있습니다.

## 환경 설정 (uv)

이 프로젝트는 [uv](https://docs.astral.sh/uv/)로 Python 환경과 의존성을 관리합니다.
`uv.lock`에 정확한 버전이 고정되어 있어 어느 머신에서든 동일한 환경이 재현됩니다.

uv 설치(최초 1회):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# Windows(PowerShell): powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

의존성 설치:

```bash
uv sync          # 실행 + 개발(주피터) 의존성 모두 설치
uv sync --no-dev # 실행 의존성만 설치
```

`uv sync`가 `.python-version`에 고정된 Python 3.13을 자동으로 내려받고
`.venv/`를 만들어 주므로 별도의 `python -m venv`나 `pip install`은 필요 없습니다.

## 실행

```bash
uv run jupyter lab           # 노트북으로 실행 (커널: 프로젝트 .venv)
uv run python crawler.py     # 스크립트로 분리한 경우
```

> `crawler.ipynb`의 첫 셀에는 예전에 사용하던 `!pip install ...`이 남아 있습니다.
> uv 환경에서는 불필요하며, 실행하면 잠긴 버전과 어긋날 수 있으니 건너뛰세요.

크롤러는 네이버 금융에 실제 요청을 보냅니다(요청 간 1초 대기, 총 수 분 소요).
이미 수집된 결과가 `news_data.json`에 있으므로 재수집은 필요할 때만 하세요.

## 의존성

| 구분 | 패키지 | 용도 |
| --- | --- | --- |
| 실행 | `requests` | 네이버 금융 뉴스 목록/기사 HTTP 요청 |
| 실행 | `beautifulsoup4` | HTML 파싱 및 본문 추출 |
| 개발 | `ipykernel` | 주피터 노트북 커널 |

패키지 추가/삭제는 pip 대신 uv로 합니다. `uv.lock`이 함께 갱신되므로
변경 사항을 커밋해 주세요.

```bash
uv add <package>          # 실행 의존성 추가
uv add --dev <package>    # 개발 의존성 추가
uv remove <package>
```
