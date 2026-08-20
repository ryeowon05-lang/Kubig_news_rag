# Dense vs Hybrid retrieval 파일럿 평가

이 평가는 50개의 source_seeded 질문으로 기존 Dense Retrieval과 Hybrid Retrieval을 비교합니다.
질문 제작용 source article은 50개지만 검색 corpus는 축소하지 않습니다. 두 시스템 모두 전체
432개 기사와 1,377개 청크에서 검색하며 기존 candidate_k, rrf_k, Chroma 설정을 그대로 사용합니다.

## 표본 선정과 배정

assign 명령은 retrieval 결과나 예상 성능을 보지 않고 고정 seed로 날짜 층화표본을 만듭니다.

| 날짜 stratum | corpus 기사 수 | 표본 | annotator_a | annotator_b |
|---|---:|---:|---:|---:|
| 2026-07-31~2026-08-05 | 65 | 10 | 5 | 5 |
| 2026-08-06 | 67 | 10 | 5 | 5 |
| 2026-08-07 | 195 | 18 | 9 | 9 |
| 2026-08-08 | 105 | 12 | 6 | 6 |

    uv run python -m evaluation.cli assign --seed 42 --output evaluation/pilot_assignment.json

각 stratum의 표본 뒤에는 같은 seed로 섞인 나머지 기사 전체가 reserve 순서로 저장됩니다.
교체는 incomplete_article, question_not_possible, near_duplicate_event 중 하나의 사유만 허용하며,
동일 stratum의 다음 reserve를 사용하고 사유를 기록해야 합니다. 교체 판단에 Dense나 Hybrid 결과를
사용하지 않습니다.

질문 유형 총량은 exact_token 13, abstract 13, multi_aspect 12, factoid 12입니다.
날짜별 배분은 각각 3/3/2/2, 3/2/3/2, 4/5/4/5, 3/3/3/3입니다. 기사 내용상 배정 유형이
불가능하면 같은 날짜 stratum 안에서 다른 기사와 유형을 교환해 전체 quota를 유지합니다. 유형은
검색 조건을 다양화하기 위한 층화 변수이며 특정 시스템에 유리한 질문을 만들기 위한 기준이 아닙니다.

## 질문 작성과 self-check

pilot_retrieval_eval.jsonl은 실제 50문항 파일이며 현재 완전히 비어 있습니다.
pilot_retrieval_eval.example.jsonl은 명백한 placeholder 한 줄만 가진 작성 예시이고 CLI가
자동으로 읽지 않습니다.

AI 초안을 받을 수 있지만 작성자가 source 기사와 대조해 질문을 확정해야 합니다.

    uv run python -m evaluation.cli generate-question 123 --category abstract --query-id pilot_001 --author annotator_a

모든 레코드는 construction_method를 source_seeded, review_mode를 ai_assisted_self_check로 사용하고
seed_article_id를 initial gold로 gold_article_ids에 포함합니다. reviewer는 null이어도 괜찮으며
가짜 교차검수자를 적지 않습니다. approved가 되려면 작성자가 다음 self_check를 모두 true로 확인합니다.

- answer_supported_by_source
- natural_question
- not_title_copy
- not_duplicate
- source_article_id_verified

AI-assisted generation과 single-annotator self-check에는 질문·gold 편향을 놓칠 수 있다는 한계가
있습니다. 교차검수를 수행한 것처럼 해석해서는 안 됩니다.

## blind pooling과 추가 gold

질문과 initial gold를 확정한 뒤에는 질문 문구를 바꾸지 않습니다. 다음 명령은 전체 corpus에서
Dense와 Hybrid를 실행하고 article_id로 합친 후보를 출력합니다. 출력에는 어느 시스템이
반환했는지와 원래 순위가 포함되지 않습니다.

    uv run python -m evaluation.cli pool "확정한 질문"
    uv run python -m evaluation.cli judge "확정한 질문"

AI 판정은 relevant, not_relevant, uncertain과 근거 문장을 반환합니다. relevant 기사는 작성자가
확인한 뒤 gold_article_ids와 evidence에 추가할 수 있고, not_relevant는 추가하지 않습니다.
uncertain은 작성자가 해결하기 전에는 final validation과 평가 실행이 거부됩니다. AI 출력은 평가
JSONL을 자동 수정하지 않습니다.

이 방식은 source 기사 하나만 정답으로 강제하는 known-item retrieval과 다릅니다. Source는 initial
gold일 뿐이며, pooled 후보 중 질문에 충분히 답하는 다른 기사도 gold가 되는 article-level relevance
평가입니다. 다른 gold를 찾기 위해 432개 전체를 사람이 읽지는 않습니다.

## 검증과 실행

작성 중 draft는 명시적으로 허용해 검증합니다.

    uv run python -m evaluation.cli validate evaluation/pilot_retrieval_eval.jsonl --allow-draft

최종 실행 전에는 모든 문항이 approved이고 uncertain이 없어야 합니다.

    uv run python -m evaluation.cli validate evaluation/pilot_retrieval_eval.jsonl
    uv run python -m evaluation.cli run evaluation/pilot_retrieval_eval.jsonl --json-output evaluation/results/pilot.json --csv-output evaluation/results/pilot.csv

실제 평가 파일이 비어 있으면 retriever를 로드하거나 결과 파일을 만들지 않습니다. 결과에는 실행 시각,
Git commit, 질문 파일 SHA-256, 전체 corpus 크기, source article 수, retrieval 설정, raw chunk 순위와
중복 제거한 article 순위가 기록됩니다. API key나 로컬 캐시 경로는 저장하지 않습니다.

지표는 기사 단위 Hit@1/3/5와 MRR@5이며 전체, 날짜 stratum별, 질문 유형별로 출력됩니다. 날짜별
요약에는 해당 stratum의 corpus 기사 수와 표본 수도 포함됩니다. 날짜별 표본이 작으므로 날짜 차이는
탐색적으로만 해석하며 통계적 우월성을 주장하지 않습니다.

## 실제 문항 작성 순서

1. assign으로 고정 배정표와 reserve를 생성합니다.
2. 각 작성자는 배정된 source 기사를 읽고 AI 초안을 참고해 질문과 initial gold를 확정합니다.
3. 다섯 가지 self-check를 직접 확인하고 질문을 동결합니다.
4. Dense/Hybrid blind pool을 만들고 AI 판정 근거를 검토합니다.
5. relevant 후보는 추가 gold로 반영하고 uncertain은 작성자가 직접 해결합니다.
6. final validation 후 전체 432개 기사 corpus에서 paired 평가를 한 번 실행합니다.
