# Data Preprocessing & EDA Documentation

이 문서는 MovieLens 100k 및 1M 데이터셋의 탐색적 데이터 분석(EDA), 정제, 그리고 Train/Val/Test 분할 파이프라인에 대한 상세 설명서입니다.

---

## 1. 데이터셋 비교 & EDA 요약

| 지표 (Metric) | MovieLens 100k (프로토타입) | MovieLens 1M (스케일업) | 비고 |
| :--- | :--- | :--- | :--- |
| **전체 평점 수** | 100,000 | **1,000,209** | 10배 확장 |
| **유저 수** | 943 | **6,040** | 6.4배 확장 |
| **영화(아이템) 수** | 1,682 | **3,883** (3,706개 평가됨) | 2.3배 확장 |
| **행렬 희소도 (Sparsity)**| 93.70% | **95.53%** | 희소성 심화 (하이브리드 접근 필요성 입증) |
| **전체 평균 평점** | 3.53 | **3.58** | 데이터셋 간 평점 분포 유사 |
| **유저당 평가 수** | 20 ~ 737 (중앙값 65) | 20 ~ 2,314 (중앙값 96) | 전형적인 롱테일(Long-tail) 불균형 |
| **콜드 아이템 (평가 $\le 5$)** | 22.8% | 8.9% | 희귀 영화 비율 존재 |

### EDA 시각화 플롯
* 100k 분포: `data/plots/eda_overview_100k.png`
* 1M 분포: `data/plots/eda_overview_1m.png`
*(평점 분포, 유저별/아이템별 롱테일 분포, 장르별 영화 수 2x2 서브플롯 제공)*

---

## 2. 데이터 품질 점검 및 정제 (Cleaning)

1. **중복 평점 및 결측치 처리**:
   - 동일 유저-영화 중복 평점은 가장 최근 타임스탬프(`timestamp`) 기록만 유지
   - 메타데이터에 영화 정보가 없는 유효하지 않은 평점 제거
2. **결정론적 정렬 (Deterministic Multi-key Sorting)**:
   - 동일 타임스탬프가 다수 존재할 때 OS 및 라이브러리 버전에 따라 분할 결과가 달라지는 문제 방지
   - `["timestamp", "user_id", "item_id"]` 복합키 정렬로 모든 팀원 환경에서 100% 동일한 분할 재현
3. **0-based 연속 정수 재색인**:
   - 원본 `user_id`, `item_id`를 `0 ... N-1`의 연속 정수(`u`, `i`)로 매핑 (`id_maps.json`)
4. **19개 장르 스키마 통일**:
   - 100k와 1M의 장르 규격을 동일한 19개 장르 컬럼(unknown 포함)으로 일치화하여 모델링 코드 호환성 보장

---

## 3. Train / Val / Test 분할 및 누수 방지

* **유저별 층화 분할 (User-stratified 7 : 1 : 2)**:
  * 모든 유저가 Train, Val, Test에 고루 분포하도록 유저별로 70% Train, 10% Val, 20% Test 분할
  * **100k**: Train 70,012건 / Val 9,988건 / Test 20,000건
  * **1M**: Train 700,243건 / Val 99,950건 / Test 200,016건
* **데이터 누수(Leakage) 차단**:
  * 글로벌 평균 평점, 유저/아이템별 평균 평점, 유저별 평가 개수($N$) 등 정규화 통계치는 **철저하게 Train 데이터만을 기준**으로 산출하여 `train_stats.npz`에 저장

---

## 4. 전처리 산출물 명세 (Artifacts)

산출물은 `data/100k/preprocessed/` (및 `data/1m/preprocessed/`)에 생성됩니다:

| 파일명 | 설명 | 사용 대상 |
| :--- | :--- | :--- |
| `train.csv`, `val.csv`, `test.csv` | 분할 평점 데이터 (`user_id`, `item_id`, `u`, `i`, `rating`, `timestamp`) | 모든 추천 모델 |
| `items.csv` | 0-based 재색인 메타데이터 (제목, 연도, 연대, 19개 장르 플래그) | `models/content_based.py` |
| `users.csv` | 0-based 재색인 유저 인포 (나이, 성별, 직업 등) | 메타/특징 분석 |
| `id_maps.json` | 원본 ID $\leftrightarrow$ 0-based 인덱스 변환 맵 | 인덱스 역추적 |
| `train_stats.npz` | Train 기준 글로벌 평균, 유저/아이템별 평균, **유저별 평가수($N$)** | `main.py` 하이브리드 결합 가중치 계산 |

> **참고**: TF-IDF 임베딩, 코사인 유사도 행렬, 희소 행렬(CSR Matrix) 등 알고리즘 관련 피처 엔지니어링은 전처리 단계에 포함하지 않으며, 각 모델 모듈(`models/content_based.py`, `models/model_based.py`)에서 직접 수행합니다.

---

## 5. 실행 방법

프로젝트 루트 디렉터리에서 실행합니다:

```bash
# MovieLens 100k 전처리 및 EDA 실행
python data/preprocess_100K.py

# MovieLens 1M 전처리 및 EDA 실행 (ml-1m 데이터 다운로드 후)
python data/preprocess_1m.py
```

