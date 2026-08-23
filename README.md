# Kr News Rag

한국 금융 뉴스 기반 RAG 에이전트 프로젝트.

뉴스 수집·정제부터 청킹과 인덱싱, Dense/BM25/Hybrid 검색,
Multi-Query·HyDE 질의 확장, Claude 답변 생성, CLI와 Streamlit UI까지 포함합니다.

## 폴더 구조

기능별 코드는 `preprocessing/`, `indexing/`, `generation/`으로 나누고,
원본과 재생성 가능한 산출물은 `data/` 아래에 둡니다.

```
app.py                              # Streamlit UI
cli.py                              # 검색 방식·질의 확장·게이트를 비교하는 CLI
rag.py                              # 검색부터 근거 기반 답변까지 연결하는 RAG 파이프라인
ui_format.py                        # CLI와 UI가 공유하는 출처 표시 형식

preprocessing/                      # 뉴스 수집 및 정제
├── crawler.ipynb                   # 네이버 금융 종목 뉴스 수집
└── clean.py                        # 중복 병합, 본문 정제, 정제 리포트 생성

indexing/                           # 청킹, 인덱싱, 검색
├── chunk.py                        # 기사 본문을 검색용 청크로 분할
├── build_index.py                  # BGE-M3 임베딩과 Chroma 인덱스 생성
├── search.py                       # Dense 검색용 단독 CLI
├── retriever.py                    # Chroma 기반 Dense retriever
├── bm25.py                         # 한국어 형태소 기반 BM25 retriever
├── hybrid.py                       # Dense와 BM25 결과의 RRF 결합
├── tokenize_ko.py                  # Kiwi 기반 한국어 토큰화
├── query_preprocess.py             # 질의 정규화와 종목명 별칭 처리
├── query_expand.py                 # Multi-Query·HyDE 질의 생성
└── expanding.py                    # 확장 질의별 검색 결과의 RRF 결합

generation/                         # Claude 답변 생성 및 검색 라우팅
├── README.md                       # 생성 단계의 입력·출력 계약
├── llm.py                          # Anthropic API 어댑터
├── prompt.py                       # 근거 인용·답변 거절 규칙 프롬프트
└── router.py                       # RETRIEVE/CHAT 검색 게이트

data/
├── raw/                            # 수집 원본 (수정하지 않음)
│   ├── news_data.json              # 크롤러 최종 산출물
│   └── crawl_cache/                # 종목별 수집 중간 저장 (git 미추적)
├── processed/                      # 정제·청킹 산출물
│   ├── news_data_clean.json        # 정제된 기사
│   ├── clean_report.json           # 정제 규칙별 적용 건수
│   └── news_chunks.jsonl           # 검색·인덱싱용 청크 (재생성 가능)
└── chroma/
    └── chroma.sqlite3              # Dense 검색 인덱스 (재생성 가능)

tests/                              # 전처리·검색·생성·CLI·UI 단위 테스트
pyproject.toml                      # 프로젝트 메타데이터와 의존성
uv.lock                             # 재현 가능한 의존성 잠금 파일
.env.example                        # 필요한 환경 변수 예시
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

Dense 검색은 임베딩 전에 Query의 Unicode와 공백을 정리하고 알려진 종목명 별칭을
정규화합니다. 예를 들어 `삼전 반도체 전망`은 `삼성전자 반도체 전망`으로,
`네이버 AI와 현대자동차 전략`은 `NAVER AI와 현대차 전략`으로 검색됩니다.
지원하는 canonical 종목명은 다음과 같습니다.

```text
삼성전자, SK하이닉스, 현대차, LG에너지솔루션, 삼성바이오로직스,
KB금융, HD현대중공업, 한화에어로스페이스, 두산에너빌리티, NAVER
```

`--device`는 `cpu`, `cuda`(NVIDIA GPU), `mps`(Apple Silicon GPU) 중에 고릅니다.
CPU로 1,377개 청크를 인덱싱하면 30분 이상 걸리므로 GPU가 있으면 지정하세요.

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

> **메타데이터 주의:** 현재 `stock_names`는 본문에서 언급된 모든 종목이 아니라
> 기사를 수집한 네이버 금융 종목 뉴스 페이지를 나타냅니다. 따라서 Query에서
> 정규화한 종목명으로 `stock_names` 필터를 적용하지 않습니다. 향후 종목 필터가
> 필요하면 `source_page_stocks`와 `mentioned_stocks`의 분리를 별도로 논의해야 합니다.

### 파이썬에서 Dense retrieval 호출

LLM 파이프라인처럼 검색을 반복 호출하는 쪽에서는 `indexing/retriever.py`를 씁니다.
Chroma 접속과 BGE-M3 로딩을 안에서 처리하므로 질문 문자열만 넘기면 됩니다.

```python
from indexing.retriever import get_retriever

for hit in get_retriever().retrieve("삼전 반도체 실적 전망", top_k=5):
    print(hit["similarity"], hit["title"], hit["date"], hit["url"])
    print(hit["content"])
```

`get_retriever()`는 설정(DB 경로, collection, 모델, device)별로 인스턴스를 재사용하므로
BGE-M3는 프로세스당 한 번만 로딩됩니다. collection과 모델은 첫 `retrieve()` 호출
시점에 지연 로딩되며, 인덱스가 없으면 모델을 내려받기 전에 `FileNotFoundError`로
끝납니다. GPU를 쓰려면 `get_retriever(device="cuda")`로 만듭니다.

반환값은 청크 하나당 dict 하나이고 similarity 내림차순입니다. `chunk_id`, `content`,
`distance`, `similarity`(= 1 - distance)와 청크 메타데이터 전체(`title`, `date`, `url`,
`stock_names`, `doc_type` 등)를 담고 있어 답변의 출처 표기에 그대로 쓸 수 있습니다.

### Hybrid 검색 (Dense + BM25)

Dense 검색은 의미를 맞추지만 정확한 토큰은 놓칩니다. 종목코드 `005930`으로 검색하면
임베딩은 그게 뭘 가리키는지 모르기 때문에 무관한 기사가 1위로 올라옵니다. BM25가 그
구멍을 메우고, 두 결과를 **RRF(Reciprocal Rank Fusion)** 로 합칩니다.

```python
from indexing.hybrid import get_hybrid_retriever

hits = get_hybrid_retriever(device="mps").retrieve("005930 관련 뉴스", top_k=5)
for h in hits:
    print(h["rrf_score"], h["dense_rank"], h["bm25_rank"], h["title"])
```

`DenseRetriever`와 같은 `retrieve(question, top_k)` 인터페이스라서 `RAGPipeline`에
그대로 넣을 수 있습니다. Dense 단독으로 쓰려면 `get_retriever()`를 그대로 쓰면 됩니다.

**RRF 동작** — 각 검색기에서 후보를 `candidate_k`(기본 20)개씩 뽑고, 청크마다
`sum(1 / (rrf_k + 순위))`를 더해 정렬합니다(`rrf_k` 기본 60). 점수가 아니라 **순위**로
합치는 이유는 코사인 유사도(~0.6)와 BM25 점수(~12)가 서로 비교할 수 없는 척도이기
때문입니다. 두 검색기가 함께 높게 본 청크가 한쪽만 좋아한 청크를 이깁니다.

> **후보 풀은 top_k보다 넓어야 합니다.** 양쪽 top-5만 합치면 융합될 여지가 거의 없습니다.

**한국어 BM25 토크나이저** — 공백으로 자르면 질문의 `삼성전자`가 본문의 `삼성전자의`와
매칭되지 않아 BM25가 아무 일도 못 합니다. `kiwipiepy`로 형태소를 분리해 조사를 떼고
어간만 남깁니다. 명사·외국어(`SL`)·숫자(`SN`)·용언 어간을 남기는데, `SL`이 특히
중요합니다 — NAVER, SK, HBM, AI가 전부 여기 걸립니다.

konlpy나 mecab을 쓰지 않은 이유는 Java나 C 라이브러리가 필요해 팀원 환경에 따라 설치가
깨지고, `uv.lock`으로 맞춰둔 재현성이 무너지기 때문입니다. kiwipiepy는 순수 wheel입니다.

BM25 인덱스는 `news_chunks.jsonl`을 읽어 메모리에 만듭니다(1,377청크 기준 약 5초).
별도 파일로 저장하지 않으므로 청크 파일과 어긋날 일이 없습니다.

> ⚠️ **Chroma 인덱스와 청크 파일은 같은 `chunk.py` 실행 결과여야 합니다.** `chunk_id`가
> 본문 해시라서 정제·청킹 규칙을 바꾸면 값이 전부 달라지고, RRF 조인이 조용히 실패합니다.
> `HybridRetriever`가 이 경우를 감지해 경고하지만, 규칙을 바꿨다면 `chunk.py` →
> `build_index.py --rebuild`를 함께 다시 실행하세요.

### 질의 확장 (Multi-Query / HyDE)

사용자가 쓴 표현은 기사에 그 답이 적혀 있는 여러 표현 중 하나일 뿐입니다. 질문을 여러
검색어로 바꿔 각각 검색한 뒤 같은 RRF로 합칩니다.

```python
from indexing.expanding import ExpandingRetriever
from indexing.hybrid import get_hybrid_retriever
from indexing.query_expand import MultiQueryExpander, HyDEExpander

retriever = ExpandingRetriever(
    base=get_hybrid_retriever(device="mps"),
    expander=MultiQueryExpander(),   # 또는 HyDEExpander()
)
hits = retriever.retrieve("왜 최근에 코스피가 떨어졌어?", top_k=5)
print(hits[0]["expanded_queries"], hits[0]["matched_queries"])
```

| 방식 | 하는 일 |
| --- | --- |
| **Multi-Query** | 질문을 뉴스 어휘로 3가지쯤 바꿔 씀. `왜 최근에 코스피가 떨어졌어?` → `코스피 하락 원인`, `코스피 급락 이유`, `코스피 지수 하락 배경` |
| **HyDE** | 답이 될 법한 가상 기사 본문을 지어내 그것으로 검색. 질문과 기사는 문체가 달라서, 기사처럼 생긴 텍스트를 임베딩하면 실제 기사에 더 가깝게 붙습니다 |

**둘 다 원본 질문을 항상 첫 번째 검색어로 넣습니다.** 재작성은 주제를 벗어날 수 있고
HyDE 본문은 아예 지어낸 것이라, 확실히 맞는 질의 하나를 융합의 안전망으로 남겨둡니다.
같은 이유로 **LLM 호출이 실패하면 예외를 던지지 않고 원본 질문만으로 검색합니다** —
확장은 최적화이지 필수 경로가 아닙니다.

> ⚠️ **HyDE가 지어낸 본문은 검색용으로만 쓰입니다.** 답변 프롬프트는 여전히 사용자의
> 원본 질문으로 만들어지므로, 지어낸 문장이 근거 자료로 모델에 들어가지 않습니다.
> `tests/test_expanding.py`가 이 성질을 테스트로 고정하고 있습니다.

`ExpandingRetriever`도 같은 `retrieve(question, top_k)` 인터페이스라 `RAGPipeline`에
그대로 들어갑니다. 후속 질문을 독립형 검색어로 재작성하는 것도 "질문 → 검색어 여러 개"
라서, 나중에 붙인다면 새 배관 없이 expander 하나만 추가하면 됩니다.

**비용** — 질문마다 LLM 호출이 1회 더 듭니다. 검색 자체(BM25·Chroma)는 몇 배로 늘어도
수십 ms 수준이라 무시할 만합니다.

### 대화와 검색 게이트

`RAGPipeline`은 이전 턴을 받아 후속 질문에 답할 수 있고, 선택적으로 **검색 게이트**를
둘 수 있습니다.

```python
from generation.router import SearchGate
from rag import RAGPipeline

pipeline = RAGPipeline(retriever=..., gate=SearchGate())
r = pipeline.ask("더 쉽게 설명해줘", history=history)
r.searched   # False — 검색 없이 대화 내용만으로 답함
```

게이트가 없으면 예전처럼 모든 턴이 검색합니다. 게이트를 붙이면 턴마다 LLM이
`RETRIEVE` / `CHAT`을 판단하고, `CHAT`이면 뉴스 자료 없이 대화 이력만으로 답합니다.
판단이 실패하거나 애매하면 **검색 쪽으로 기웁니다** — 사실 질문에 근거 없이 답하는 것이
불필요하게 검색하는 것보다 나쁜 실패라서.

**왜 유사도 임계값이 아니라 LLM인가** — 처음엔 임계값을 재봤는데 이 코퍼스에서는 두 부류가
안 갈립니다.

| 질의 | 유사도 |
| --- | --- |
| `오늘 서울 날씨 어때` (잡담) | **0.558** |
| `삼성바이오로직스 위탁생산` (정상) | 0.552 |
| `005930` (정상) | **0.428** |

잡담이 정상 질문보다 높고, 종목코드 질의는 전체 최저입니다(임베딩이 숫자열을 모름).
BM25 점수를 같이 봐도 최선 조합에서 잡담 6/14가 통과합니다. 스칼라 컷으로는 못 가릅니다.

> ⚠️ **후속 질문의 한계** — 질문에 종목명이 있으면 잘 검색됩니다(`그럼 SK하이닉스는?`
> 유사도 0.629). 반면 대명사만 있으면 실패합니다: `거기는 어때?` 0.481, `그 회사 실적은?`
> 0.585 모두 무관한 기사를 가져옵니다. 후속 질문을 독립형 검색어로 재작성하는 단계를
> 넣으면 해결되며, 그건 "질문 → 검색어" 형태라 `indexing/query_expand.py`에 expander를
> 하나 더 추가하는 것으로 붙습니다.

### CLI로 질의하기

브라우저 없이 터미널에서 같은 파이프라인을 씁니다. Streamlit 사이드바의 옵션이
그대로 플래그로 대응합니다.

```bash
uv run python cli.py "삼성전자 HBM 실적 전망은?"                  # 기본: Hybrid
uv run python cli.py "005930 주가" --mode bm25                    # BM25만
uv run python cli.py "HBM 수요" --mode dense --expand hyde        # Dense + HyDE
uv run python cli.py "코스피 시황" --retrieval-only               # 검색만, API 크레딧 0
uv run python cli.py --chat --gate                                # 대화 + 검색 게이트
uv run python cli.py "환율 전망" --json                           # 기계 판독용 출력
```

| 플래그 | 값 (기본값) | 뜻 |
| --- | --- | --- |
| `--compare` | 조합 목록 | 비교할 조합을 직접 지정 (`--mode`/`--expand`보다 우선) |
| `--mode` | `dense` / `bm25` / **`hybrid`** | 검색 방식, 쉼표로 여러 개 |
| `--expand` | **`none`** / `multi_query` / `hyde` | 질의 확장, 쉼표로 여러 개 |
| `--gate` | 플래그 (꺼짐) | 검색이 필요한 질문인지 LLM이 판단 |
| `--chat` | 플래그 (꺼짐) | 이력을 기억하는 대화 모드 |
| `--retrieval-only` | 플래그 (꺼짐) | Claude를 호출하지 않음 — **API 키 없이 동작** |
| `--top-k` | 정수 (**5**) | 조합마다 가져올 청크 수 |
| `--candidate-k` | 정수 (**20**) | RRF로 융합하기 전 각 검색기가 뽑는 후보 수 |
| `--rrf-k` | 정수 (**60**) | RRF 상수 `1/(k+rank)` — 키우면 순위 차이가 완만해집니다 |
| `--device` | **`auto`** / `cpu` / `cuda` / `mps` | 임베딩 연산 장치 (`auto`는 cuda → mps → cpu 순으로 탐지) |
| `--json` | 플래그 (꺼짐) | 결과를 JSON으로 출력 (조합별 `runs` 배열) |

인덱스가 없거나 `ANTHROPIC_API_KEY`가 없으면 스택 트레이스 대신 `오류: ...` 한 줄을
stderr로 내고 종료 코드 1을 돌려줍니다.

출력은 `출처 → 구분선 → 답변` 순입니다.

```
출처 2건
  [1] [칩톡]"살 빼야 출하된다"…AI 공룡들, 메모리 기근에 강제 'HBM...
      2026.08.08 08:00 | 삼성전자 · SK하이닉스
      RRF 0.0325 · dense #1 · bm25 #2 · 유사도 0.7140 · BM25 9.88 · 인용 [뉴스1], [뉴스2]
      https://finance.naver.com/item/news_read.naver?article_id=0005800438&...
      것으로 전망한다"며 "올해 하반기 기준 HBM4 매출이 삼성전자 전체 HBM 매출의…
  [2] "열 받으면 안돼"…AI시대 "발열 줄이고 빨리 식히기" 경쟁
      ...
────────────────────────────────────────────────────────────
삼성전자의 HBM 관련 실적 전망은 다음과 같습니다.
- 김재준 삼성전자 메모리전략마케팅실장 부사장은 …[뉴스1][뉴스2].
```

검색은 즉시 끝나므로 **출처가 답변보다 먼저** 뜨고, 답변은 Claude가 쓰는 대로
스트리밍됩니다. 출처 줄의 `RRF 0.0325 · dense #1 · bm25 #2`는 어느 검색기가 몇 위로
이 청크를 올렸는지, `인용 [뉴스1], [뉴스2]`는 답변 속 어느 번호가 이 기사를 가리키는지를
뜻합니다(같은 기사의 청크 두 개가 top-k에 들어오면 하나로 합칩니다).

리트리버를 튜닝할 때는 `--retrieval-only`를 쓰세요. 답변 생성은 질문마다 Claude API
비용이 듭니다.

#### 대화 모드

```bash
uv run python cli.py --chat --gate
```

`질문>` 프롬프트가 뜨고, 매 턴이 이전 턴을 기억하므로 후속 질문이 됩니다.
`exit` / `quit` / `q` / `종료` / `나가기` 중 아무거나 입력하거나 Ctrl-D로 끝냅니다.
`--gate`를 켜면 잡담이나 직전 답변에 대한 재질문은 검색을 건너뛰고 `출처 없음`으로
표시됩니다.

거절되는 조합이 둘 있습니다.

- `--chat --retrieval-only` — 답변을 만들지 않으면 기억할 이력도 없습니다. 단발 질문을 쓰세요.
- `--chat`에 조합을 여러 개 — 답이 여러 개면 어느 답을 이력에 남길지 정할 수 없습니다.

#### 여러 기법 한 번에 비교

`--compare`에 조합을 직접 나열하면 한 질문을 각 조합으로 돌려 나란히 보여줍니다.
조합 이름은 표에 찍히는 이름과 같은 표기(`hybrid`, `dense+hyde`)를 씁니다.

```bash
uv run python cli.py "HBM 수요 전망" \
  --compare dense,hybrid,dense+hyde,dense+multi_query --retrieval-only
```

```
변형별 순위 (· = 그 변형은 못 찾음)
dense  hybrid  dense+hyde  dense+multi_query  기사
    1       2           1                  2  [칩톡]"살 빼야 출하된다"…AI 공룡들…
    2       1           3                  1  삼전·하닉은 월가 'ATM'이었다?...
    ·       4           5                  5  엔비디아, HBM 공급부족 대응…
    ·       5           ·                  4  메모리 품귀에…엔비디아 HBM4E 용량 축소 검토
```

숫자는 그 조합에서의 순위, `·`는 그 조합이 못 찾았다는 뜻입니다. 모든 조합이
찾은 기사가 위로 올라오므로, 아래쪽 `·`가 섞인 줄이 기법 간 차이가 나는 지점입니다.

`--mode`/`--expand`에 쉼표 목록을 주면 모든 짝을 훑습니다 — 조합을 일일이
적기보다 전수 비교를 하고 싶을 때 씁니다.

```bash
uv run python cli.py "HBM 수요 전망" --mode dense,bm25,hybrid --retrieval-only
uv run python cli.py "HBM 수요 전망" --mode hybrid --expand none,multi_query,hyde
```

무거운 모델은 조합 사이에 공유되므로 BGE-M3는 몇 개를 비교하든 한 번만 올라옵니다.
`--json`을 붙이면 조합별 결과가 `runs` 배열로 나옵니다.

```json
{
  "question": "HBM 수요 전망",
  "runs": [
    {"variant": "dense",  "mode": "dense",  "expand": "none", "answer": "...", "sources": [...]},
    {"variant": "hybrid", "mode": "hybrid", "expand": "none", "answer": "...", "sources": [...]}
  ]
}
```

`--retrieval-only`와 함께 쓰면 `answer`는 `null`이고 `sources`만 채워집니다.

> 확장(`hyde`, `multi_query`)은 조합마다 Claude를 한 번씩 더 부릅니다. `--retrieval-only`
> 로도 확장은 LLM을 쓰므로, 확장을 비교할 때는 API 키가 필요합니다.

### Streamlit UI

```bash
uv run streamlit run app.py
```

- **UI 모드**: `단일 질문` / `대화` 전환
  - **단일 질문** — 매 질문이 독립적입니다. 검색 설정을 바꿔가며 비교할 때 쓰세요
  - **대화** — 이력을 기억해 후속 질문이 됩니다. 검색 게이트를 켜고 끌 수 있고, 답변마다
    검색을 했는지 표시합니다
- **검색 방식**: Hybrid / Dense only 전환 — 같은 질문으로 두 방식을 비교할 수 있습니다
- **검색만 (LLM 호출 안 함)**: 검색 결과만 봅니다. 리트리버를 튜닝할 때는 이 모드를
  쓰세요. 답변 생성은 질문마다 Claude API 비용이 듭니다
- 출처 카드마다 `RRF 0.0164 · dense #1 · bm25 —` 형태로 **어느 검색기가 이 청크를
  올렸는지** 표시합니다
- `ANTHROPIC_API_KEY`가 없으면 검색 전용으로 동작합니다

크롤러는 `data/raw/`에, 정제 스크립트는 `data/processed/`에 결과를 씁니다
(폴더는 자동 생성). 두 코드 모두 `pyproject.toml`이 있는 폴더를 프로젝트
루트로 잡으므로, 어느 디렉터리에서 실행하든 같은 `data/`를 바라봅니다.

> uv 환경에서는 불필요하며, 실행하면 잠긴 버전과 어긋날 수 있으니 건너뛰세요.

크롤러는 네이버 금융에 실제 요청을 보냅니다(요청 간 1초 대기, 총 20~30분 소요).
이미 수집된 결과가 `data/raw/news_data.json`에 있으므로 재수집은 필요할 때만 하세요.

## Retrieval evaluation 결과

최종 평가는 freeze된 source-seeded 질문 50개를 사용해 Dense와 Hybrid를 같은 질문에서
paired 방식으로 비교했습니다. 두 시스템의 blind candidate를 합친 뒤 1,299개의 고유
query–candidate pair에 대해 relevance adjudication을 수행했습니다. Final judgment가
`relevant`인 기사만 human gold에 추가했으며 `uncertain`은 relevant로 인정하지 않았습니다.

| Metric | Dense | Hybrid | Difference |
| --- | ---: | ---: | ---: |
| Hit@1 | 0.7800 | 0.9000 | +0.1200 |
| Hit@3 | 0.9200 | 0.9600 | +0.0400 |
| Hit@5 | 0.9200 | 0.9600 | +0.0400 |
| MRR@5 | 0.8333 | 0.9233 | +0.0900 |

Hybrid는 모든 frozen aggregate metric에서 Dense보다 높았습니다. 가장 큰 개선은 Hit@1과
MRR@5에서 나타났으며, 이는 Hybrid의 주된 이점이 관련 문서를 top rank로 재정렬하는 데
있음을 보여줍니다. Hit@5 개선은 `K009`와 `R015` 두 query에서 발생했습니다.
`multi_aspect`와 `2026-08-07` subgroup에서도 Hybrid가 더 높았지만, subgroup 표본이 작으므로
이 결과는 탐색적으로만 해석합니다.

이 평가는 query 50개의 소규모 평가이며 source-seeded, AI-assisted 질문과 relevance judgment를
사용했습니다. Pass1과 review는 동일 모델이 수행했고 독립적인 human cross-review는 없었습니다.
사전 정의된 confidence interval이나 통계검정도 없으므로, 더 넓은 query 모집단에서 Hybrid가
통계적으로 우월하다고 일반화할 수 없습니다.

Frozen input과 기존 judgment를 사용해 결과를 다시 생성하는 명령은 다음과 같습니다. 이 명령은
OpenAI API를 호출하지 않습니다.

```bash
uv run python -m evaluation.final_evaluation
```

- [최종 aggregate 및 subgroup metrics](evaluation/results/retrieval_eval_50_metrics.json)
- [Query별 paired comparison](evaluation/results/retrieval_eval_50_query_comparison.csv)
- [평가 설정, judgment 진단 및 무결성 manifest](evaluation/results/retrieval_eval_50_manifest.json)

## 의존성

| 구분 | 패키지 | 용도 |
| --- | --- | --- |
| 실행 | `requests` | 네이버 금융 뉴스 목록/기사 HTTP 요청 |
| 실행 | `beautifulsoup4` | HTML 파싱 및 본문 추출 |
| 실행 | `sentence-transformers` | BGE-M3 임베딩 |
| 실행 | `chromadb` | 벡터 인덱스 저장·검색 |
| 실행 | `kiwipiepy` | 한국어 형태소 분석 (BM25 토크나이저) |
| 실행 | `rank-bm25` | BM25 키워드 검색 |
| 실행 | `anthropic` | Claude API 답변 생성 |
| 실행 | `streamlit` | 데모 UI |
| 개발 | `ipykernel` | 주피터 노트북 커널 |
| 개발 | `pytest` | 자동 테스트 |

패키지 추가/삭제는 pip 대신 uv로 합니다. `uv.lock`이 함께 갱신되므로
변경 사항을 커밋해 주세요.

```bash
uv add <package>          # 실행 의존성 추가
uv add --dev <package>    # 개발 의존성 추가
uv remove <package>
```
