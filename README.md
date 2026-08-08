# Kr News Rag

한국 금융 뉴스 기반 RAG 에이전트 프로젝트.

현재는 1주차 산출물인 데이터 준비 단계(수집 + 정제)와 그 결과 데이터가
포함되어 있습니다.

## 폴더 구조

전처리 코드는 `preprocessing/`, 데이터는 `data/` 아래로 나눠 둡니다.

```
preprocessing/                  # 데이터 준비 단계 코드
├── crawler.ipynb               # 네이버 금융 종목 뉴스 수집
└── clean.py                    # 수집 결과 정제

data/
├── raw/                        # 수집 원본 (수정하지 않음)
│   ├── news_data.json          # 크롤러 산출물
│   └── crawl_cache/            # 종목별 중간 저장 (git 미추적)
└── processed/                  # 정제 결과
    ├── news_data_clean.json    # clean.py 산출물
    └── clean_report.json       # 정제 규칙별 적용 건수
```

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
uv run jupyter lab                      # 크롤러 노트북 실행 (커널: 프로젝트 .venv)
uv run python preprocessing/clean.py    # 수집 결과 정제
```

### 청킹, 인덱싱, Dense 검색

```bash
uv run python indexing/chunk.py
uv run python indexing/build_index.py --rebuild --batch-size 8 --device cpu
uv run python indexing/search.py "삼성전자 반도체 실적 전망" --top-k 5
```

임베딩 모델은 `BAAI/bge-m3`를 사용합니다. 최초 실행 시 모델 다운로드가 크고 오래
걸릴 수 있습니다. 생성되는 `data/processed/news_chunks.jsonl`과 `data/chroma/`는
재생성 가능한 대용량 산출물이므로 Git에서 제외합니다.

인덱싱이 중단되면 `--rebuild` 없이 같은 명령을 다시 실행할 수 있습니다. 결정적인
청크 ID와 Chroma `upsert`를 사용하므로 중복 레코드는 생기지 않지만, 현재 구현은
기존 레코드를 건너뛰지 않고 전체 청크를 다시 임베딩합니다. `--rebuild`를 지정하면
기존 collection을 삭제한 뒤 처음부터 생성합니다.

각 JSONL 청크는 결정적인 `chunk_id`, `article_id`, `chunk_index`, `title`, 원문
`content`, 제목과 본문을 합친 `embedding_text`, `date`, `url`, `stock_names`,
`stock_codes`, `source_ids`, `doc_type`을 가집니다. Chroma에는 세 리스트 필드를 JSON
문자열로 저장하고 검색 시 다시 리스트로 복원합니다.

크롤러는 `data/raw/`에, 정제 스크립트는 `data/processed/`에 결과를 씁니다
(폴더는 자동 생성). 두 코드 모두 `pyproject.toml`이 있는 폴더를 프로젝트
루트로 잡으므로, 어느 디렉터리에서 실행하든 같은 `data/`를 바라봅니다.

> uv 환경에서는 불필요하며, 실행하면 잠긴 버전과 어긋날 수 있으니 건너뛰세요.

크롤러는 네이버 금융에 실제 요청을 보냅니다(요청 간 1초 대기, 총 20~30분 소요).
이미 수집된 결과가 `data/raw/news_data.json`에 있으므로 재수집은 필요할 때만 하세요.

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
