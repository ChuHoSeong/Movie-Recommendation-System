# Hybrid Movie Recommendation System (Team 5)

> A personalized hybrid recommendation system that dynamically adjusts model weights based on individual user rating counts to address cold-start and data sparsity issues.

---

## 1. Project Overview

* **Course**: Machine Learning Term Project
* **Objective**: Overcome rating sparsity and cold-start limitations in movie recommendation by combining content metadata and latent factor modeling.
* **Core Strategy**:
  * **Light Users** (low $N$): Prioritize **Content-Based Filtering (TF-IDF)** using item metadata (genres, titles).
  * **Heavy Users** (high $N$): Prioritize **Model-Based Collaborative Filtering (SVD / MF)** to exploit learned latent factors.
  * **Dual Scale Verification**: Rapid prototyping with **MovieLens 100k** followed by scale-up validation on **MovieLens 1M**.

---

## 2. Team Members & Responsibilities

| Student ID | Name | Role | Responsibilities & Code |
| :--- | :--- | :--- | :--- |
| **202334556** | **Hoseong Chu** | Pipeline & Evaluation | Hybrid assembly, evaluation, project coordination (`main.py`) |
| **202239866** | **Wonjun Kim** | Data Preprocessing | Data exploration (EDA), cleaning, and train/test split (`preprocess.py`, `preprocess_1m.py`) |
| **202135594** | **Jeonghyeok Han** | Content-Based Model | Metadata feature extraction and TF-IDF recommender (`models/content_based.py`) |
| **202235149** | **Gyeonghoe Heo** | Model-Based CF | Latent factor modeling using SVD / Matrix Factorization (`models/model_based.py`) |

---

## 3. Datasets

* **Source**: [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
* **MovieLens 100k (Prototyping)**:
  * `u.data`: 100,000 ratings from 943 users across 1,682 movies.
  * `u.item`: Movie titles, release dates, and 19 genre binary flags.
  * `u.user`: User demographic information.
* **MovieLens 1M (Scale-up Validation)**:
  * `ratings.dat`: 1,000,209 ratings from 6,040 users across 3,706 rated movies (`::` separated).
  * `movies.dat`: 3,883 movies with titles and pipe-separated genres (`Action|Adventure|...`). No release date column (the year is parsed from the title).
  * `users.dat`: User demographics (`UserID::Gender::Age::Occupation::Zip`).

---

## 4. System Workflow

```
User Data
    │
    ├─► [Content-Based (TF-IDF)] ─────────┐
    │                                     ▼
    ├─► [Rating Count (N)] ───────► [Hybrid Assemble] ──► Top-K Recommendations
    │                                     ▲
    └─► [Model-Based CF (SVD)] ───────────┘
```

* **Score Formula**:
  $$\text{Score}(u, i) = w_{\text{CB}}(N_u) \cdot \widehat{R}_{\text{CB}}(u, i) + w_{\text{CF}}(N_u) \cdot \widehat{R}_{\text{CF}}(u, i)$$
  *(Scores are normalized before assembly to align output scales.)*

---

## 5. Project Structure

```
term_project/
├── data/
│   ├── raw/                 # Raw MovieLens 100k files
│   ├── preprocessed/        # Cleaned 100k datasets
│   ├── preprocessed_1m/     # Cleaned 1M datasets
│   ├── preprocessing.py     # 100k preprocessing pipeline
│   └── preprocessing_1m.py  # 1M preprocessing pipeline
├── ml-1m/                   # Raw MovieLens 1M files (ratings.dat, movies.dat, etc.)
├── models/
│   ├── content_based.py     # Content-based recommendation module (TF-IDF)
│   └── model_based.py       # Collaborative filtering module (SVD / MF)
├── plots/
│   ├── eda_overview.png     # 100k EDA distribution plots
│   └── eda_overview_1m.png  # 1M EDA distribution plots
├── preprocess.py            # Root runner for 100k
├── preprocess_1m.py         # Root runner for 1M
├── main.py                  # End-to-end execution and hybrid evaluation
├── .gitignore
└── README.md
```

---

## 6. Data Preprocessing & EDA Details

### [1] Dataset Comparison & EDA Summary

| Metric | MovieLens 100k | MovieLens 1M (Scaling) |
| :--- | :--- | :--- |
| **Total Ratings** | 100,000 | **1,000,209** |
| **Users** | 943 | **6,040** |
| **Movies (Items)** | 1,682 | **3,883** (3,706 rated) |
| **Matrix Sparsity** | 93.7% | **95.53%** (more severe sparsity) |
| **Overall Mean Rating** | 3.53 | **3.582** |
| **Train Global Mean** | 3.5292 | **3.5823** |
| **User Rating Range** | 20 ~ 737 (median 65) | 20 ~ 2,314 (median 96) |
| **Item Rating Range** | 1 ~ 583 (median 27) | 1 ~ 3,428 (median 123.5) |
| **Cold Item ($\le 5$ ratings)** | 22.8% | 8.9% |

*Sparsity, rating ranges, medians and cold-item ratios are computed over rated movies only (1,682 for 100k, 3,706 for 1M).*

![100k EDA Overview](plots/eda_overview.png)
![1M EDA Overview](plots/eda_overview_1m.png)

### [2] Quality Checks & Reproducibility
- **Cleaning**: 0 missing ratings and 0 duplicate (user, item) pairs in both datasets. The pipeline still keeps only the latest rating if duplicates ever appear.
- **Missing metadata**: 100k has one movie without a release date (item 267, title `unknown`), kept with `unknown` genre and `decadeunk` tokens. 1M has no release date column, so the year is parsed from the title only.
- **Outliers**: Extreme users (e.g. 2,314 ratings) and movies with a single rating are **kept**, because this imbalance is exactly the sparsity / cold-start problem the project targets.
- **Deterministic Multi-key Sorting**:
  - Both pipelines sort on `["timestamp", "user_id", "item_id"]` and then on `["user_id", "item_id"]` before splitting, so tied timestamps no longer change the row order.
  - This removes the NumPy-version dependence that produced different splits for the same seed. The same split was confirmed on two different machines.
- **Continuous Re-indexing**:
  - Raw `user_id` and `item_id` are re-indexed to contiguous $0 \dots N-1$ integers (`id_maps.json`).

### [3] Train / Val / Test Split & Leakage Prevention
- **Per-user Split (7 : 1 : 2)**: each user's ratings are shuffled and divided, so every user appears in all three sets.
  - **100k**: Train 70,012 / Val 9,988 / Test 20,000 (Test items unseen in Train: 0.18%).
  - **1M**: Train 700,243 / Val 99,950 / Test 200,016 (Test items unseen in Train: **0.03%**).
- **Leakage Prevention**: All normalization baselines (global mean, user/item means, user counts $N$) are computed **strictly on Train splits** (`train_stats.npz`).

### [4] Preprocessed Artifacts Specification

The exact same artifact schema is produced for both datasets (`data/preprocessed/` and `data/preprocessed_1m/`):

| File | Description | Target Module |
| :--- | :--- | :--- |
| `train.csv`, `val.csv`, `test.csv` | Split rating data (`u`, `i`, `rating`, `timestamp`) | Common / All Models |
| `train_given{3, 5, 10, 20}.csv` | **Cold-Start simulation sets** (max $k$ ratings per user) | Hybrid weighting evaluation |
| `items.csv`, `users.csv` | Re-indexed metadata tables | Feature engineering |
| `item_tfidf.npz` | Genre (3x weight) + Decade + Title TF-IDF sparse matrix | `models/content_based.py` |
| `item_item_cosine.npy` | Precomputed cosine similarity matrix (float32) | `models/content_based.py` |
| `R_train.npz` | Train rating sparse CSR matrix | `models/model_based.py` |
| `train_stats.npz` | Global mean, user/item means, and **user rating counts ($N$)** | Normalization & Hybrid assemble |
| `id_maps.json` | Mapping dictionary between raw IDs and 0-based indices | ID recovery |
| `summary.json` | Dataset statistics and verification diagnostics | Evaluation & reporting |

**Notes for modeling**
- `items.csv` has the same 19 genre flag columns (`unknown` + 18 genres) for both datasets. In 1M, `unknown` is 1 only for movies that have no genre (almost none), so it is nearly all zeros.
- Tune hyperparameters and the hybrid weight function $w(N)$ on `val.csv`; use `test.csv` only for the final evaluation.
- The light / medium / heavy user thresholds are not fixed in preprocessing; define them at the modeling stage from `user_count` in `train_stats.npz`.
- `train_given{k}.csv` keeps at most $k$ training ratings per user, so performance can be reported as a function of $N$ (the original data has no true light users, since every user has at least 20 ratings).

---

## 7. Getting Started

### 1) Run Preprocessing (100k)
```bash
python preprocess.py
# or
python data/preprocessing.py
```

### 2) Run Preprocessing (1M Scale-Up)
```bash
python preprocess_1m.py
# or
python data/preprocessing_1m.py
```

Default paths: `--data_dir ml-1m`, `--out_dir data/preprocessed_1m`, `--plot_dir plots`.

### Sanity check (expected values in `summary.json`)
Because the split is deterministic, every team member should get the same numbers. If they differ, the environment or the raw data is different.

| | 100k | 1M |
| :--- | :--- | :--- |
| `train_size` / `val_size` / `test_size` | 70,012 / 9,988 / 20,000 | 700,243 / 99,950 / 200,016 |
| `test_items_unseen_in_train_pct` | 0.18 | 0.03 |
| `train_global_mean` | 3.5292 | 3.5823 |

### 3) Run Models & Hybrid Recommender
```bash
python main.py
```
