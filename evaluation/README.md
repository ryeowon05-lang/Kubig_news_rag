# Dense vs Hybrid Retrieval 파일럿 평가

이 디렉터리는 사람이 만든 질문과 gold article로 기존 Dense Retrieval과 Hybrid
Retrieval을 비교합니다. 파일럿 목표는 8문항으로 평가 제작 방식의 실현 가능성,
질문의 자연스러움, gold 명확성, 라벨링 시간을 확인하는 것입니다. 표본이 작으므로
통계적 우월성이나 일반적인 시스템 성능을 주장할 수 없습니다.

## 질문 제작 방식

`source_seeded`는 전체 정제 corpus에서 retrieval score를 보지 않고 기사를 먼저
고른 뒤, 사람이 해당 기사로 답할 수 있는 자연스러운 질문을 씁니다. 제목이나 본문
문장을 그대로 복사하면 lexical overlap으로 평가가 쉬워지는 누수가 생기므로 피합니다.
선택 기사는 initial gold이며, 검색 결과에서 발견한 다른 정답 기사는 교차검수 후
gold에 추가할 수 있습니다.

`query_first`는 기사를 먼저 보지 않고 실제 사용자처럼 질문을 고정합니다. 그 후
Dense와 Hybrid의 서로 다른 기사 top-20을 pooling해 사람이 relevance를 판정합니다.
검색 결과를 본 뒤 질문 문구를 바꾸지 않습니다. `--blind`를 사용하면 판정 중 시스템
이름과 순위를 숨길 수 있습니다.

두 방식 모두 `data/processed/news_data_clean.json`의 전체 432개 기사와 기존 전체
청크 인덱스를 검색합니다. source-seeded 기사를 별도 corpus로 만들지 않습니다.

## 파일과 검수

- `pilot_retrieval_eval.jsonl`: 실제 사람이 작성하는 평가 파일. 처음에는 완전히 비어 있습니다.
- `pilot_retrieval_eval.example.jsonl`: 형식 설명용 draft placeholder. 실제 corpus에 없는
  음수 ID를 쓰므로 검증에 실패하며, CLI는 이 파일을 자동으로 찾거나 포함하지 않습니다.

각 레코드는 example 파일과 같은 필드를 가집니다. `category`는 `exact_token`, `entity`,
`semantic_paraphrase`, `multi_entity`, `temporal_numeric` 중 하나여야 합니다. `evidence`의
`relevance`는 `relevant`, `not_relevant`, `uncertain` 중 하나입니다.

파일럿 분담은 다음과 같습니다.

- `annotator_a`: source_seeded 2문항, query_first 2문항 작성
- `annotator_b`: source_seeded 2문항, query_first 2문항 작성
- 각자 상대방의 4문항을 검수하고 `reviewer`, `review_status`, gold와 evidence를 확정

실명은 코드에 넣지 않고 각 레코드의 `author`, `reviewer`에 기록합니다. 승인 시 두 값은
달라야 합니다. 파일럿 작성 중에는 draft와 uncertain을 허용하지만 final 실행에서는
모두 해결되어야 합니다.

## 기사 탐색과 query-first pooling

기사 탐색은 원본·정제 파일을 수정하지 않으며 retrieval score를 표시하지 않습니다.

```powershell
uv run python -m evaluation.cli articles get 123
uv run python -m evaluation.cli articles search "HBM" --limit 10
uv run python -m evaluation.cli articles sample --count 5 --seed 42
```

query-first 질문은 먼저 문구를 확정한 다음 pooled 후보를 만듭니다. 기본 pool은 각
시스템의 서로 다른 기사 top-20입니다. 청크는 기본 100개를 요청한 뒤 기사 ID로
중복 제거합니다. 이 `chunk_depth=100`은 서로 다른 기사 수를 확보하기 위한 평가 구현
세부사항이며 Hybrid의 `candidate_k=20`, `rrf_k=60`이나 RRF 공식을 튜닝하지 않습니다.

```powershell
uv run python -m evaluation.cli pool "사용자가 먼저 작성한 질문" --blind
```

blind 출력의 `candidate_id`를 기준으로 검토하되, 실제 JSONL evidence에는 확인한
`article_id`, 제목, support, relevance를 기록합니다. 도구는 질문이나 gold를 자동으로
수정하지 않습니다.

## 검증과 평가 실행

작성 중에는 draft를 명시적으로 허용합니다.

```powershell
uv run python -m evaluation.cli validate evaluation/pilot_retrieval_eval.jsonl --allow-draft
```

교차검수 후 final 검증과 실행에서는 `--allow-draft`를 빼야 합니다.

```powershell
uv run python -m evaluation.cli validate evaluation/pilot_retrieval_eval.jsonl
uv run python -m evaluation.cli run evaluation/pilot_retrieval_eval.jsonl `
  --json-output evaluation/results/pilot.json `
  --csv-output evaluation/results/pilot.csv
```

현재 실제 평가 파일은 비어 있으므로 `run`은 모델이나 인덱스를 로드하지 않고 결과도
생성하지 않습니다. 실제 실행에는 기존 `data/processed/news_chunks.jsonl`, Chroma DB,
BGE-M3가 필요합니다. 결과에는 실행 시각, Git commit, 질문 파일 SHA-256, corpus 수,
retrieval 설정, 문항·시스템별 raw 청크 및 unique 기사 순위, 지표, latency와 error가
저장됩니다. API key, 환경변수 값, 로컬 경로, 모델 캐시 경로는 저장하지 않습니다.

## 평가 단위와 지표

Retriever는 청크를 반환하지만 primary 평가는 기사 단위입니다. 겹치는 청크 때문에
한 기사가 여러 순위를 차지할 수 있으므로 충분한 청크를 먼저 받은 뒤 `article_id`로
중복 제거하고 최초 등장 순위를 보존합니다. 청크 top-5를 먼저 자르지 않습니다. raw
청크 순위와 기사별 `chunk_ids`는 진단용으로 남기지만, 사람이 gold chunk를 라벨링하지
않았으므로 chunk-level Recall이나 성능 우열을 주장하지 않습니다.

- Hit@k(출력상 Recall@k 별칭 가능): 상위 k개의 서로 다른 기사 중 하나 이상이
  `gold_article_ids`에 포함되면 문항 값 1, 아니면 0이며 이를 전체 문항에서 평균
- MRR@5: 최초 relevant 기사가 5위 이내이면 그 순위의 역수, 없으면 0
- top-5 overlap: Dense와 Hybrid의 서로 다른 기사 top-5 교집합 크기

construction method별 평균은 2~4문항에 불과해 매우 불안정합니다. 성능 차이가 크게
보이는 방식을 자동 선택하지 말고, 사람이 질문 자연스러움, gold 명확성, 라벨링 시간,
누수·편향 통제 가능성을 차례로 검토해 최종 방식을 정합니다.

## 다음 문항 추가 절차

1. 두 annotator가 배정에 따라 각 4문항을 `pilot_retrieval_eval.jsonl`에 draft로 작성합니다.
2. source-seeded는 score 없는 `articles sample/get/search`만 사용합니다.
3. query-first는 질문을 먼저 고정한 뒤 blind pool을 읽고 evidence와 gold를 기록합니다.
4. `validate ... --allow-draft`로 형식과 corpus ID를 확인합니다.
5. 상대방 문항을 교차검수하고 uncertain을 해소한 뒤 approved로 변경합니다.
6. final validate 후 paired 평가를 한 번 실행합니다.
7. 문항별 결과와 기술통계만 검토하여 최종 질문 제작 방식을 사람이 결정합니다.

최종 방식을 확정한 뒤에는 50문항을 두 사람이 25개씩 작성하고, 상대방의 25문항을
교차검수합니다. 파일럿 결과나 모델 점수를 보고 질문을 수정·선별하지 않습니다.
