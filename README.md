# Hybrid Movie Recommendation System (Team 5)

> A personalized hybrid recommendation system that dynamically adjusts model weights based on individual user rating counts ($N$) to address cold-start and data sparsity issues.

---

## 1. Project Overview

* **Course**: Machine Learning Term Project
* **Objective**: Overcome rating sparsity and cold-start limitations in movie recommendation by combining content metadata and latent factor modeling.
* **Core Strategy**:
  * **Light Users** (low $N$): Prioritize **Content-Based Filtering (TF-IDF)** using item metadata (genres, titles).
  * **Heavy Users** (high $N$): Prioritize **Model-Based Collaborative Filtering (SVD / MF)** to exploit learned latent factors.

---

## 2. Team Members & Responsibilities

| Student ID | Name | Role | Responsibilities & Code |
| :--- | :--- | :--- | :--- |
| **202334556** | **Hoseong Chu** | Pipeline & Evaluation | Hybrid assembly, evaluation, project coordination (`main.py`) |
| **202239866** | **Wonjun Kim** | Data Preprocessing | Data exploration (EDA), cleaning, and train/test split (`data/`) |
| **202135594** | **Jeonghyeok Han** | Content-Based Model | Metadata feature extraction and TF-IDF recommender (`models/content_based/`) |
| **202235149** | **Gyeonghoe Heo** | Model-Based CF | Latent factor modeling using SVD / Matrix Factorization (`models/model_based/`) |

---

## 📊 3. Dataset

* **Dataset**: [MovieLens](https://grouplens.org/datasets/movielens/) (`100k` for prototyping, `1M` for scaling)
* **Raw Files**:
  * `ratings.dat`: User ratings and interactions.
  * `movies.dat`: Movie titles, genres, and metadata.

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
│   ├── 100k/
│   │   ├── raw/                     # Raw MovieLens 100k files
│   │   └── preprocessed/            # Cleaned 100k datasets (train, val, test, ...)
│   ├── 1m/
│   │   ├── raw/                     # Raw MovieLens 1M files
│   │   ├── preprocessed/            # Cleaned 1M datasets
│   │   └── download.py              # 1M dataset auto-downloader
│   ├── plots/                       # 100k & 1M EDA distribution plots
│   ├── preprocess_100k.py           # 100k Preprocessing & EDA pipeline
│   ├── preprocess_1m.py             # 1M Preprocessing & EDA pipeline
│   └── doc.md                       # Preprocessing & EDA documentation
├── models/
│   ├── content_based/
│   │   ├── plots/                   # CB model evaluation plots
│   │   ├── content_based.py         # Content-based recommendation module
│   │   └── doc.md                   # CB model documentation
│   └── model_based/
│       ├── plots/                   # CF model evaluation plots
│       ├── model_based.py           # Collaborative filtering module (SVD / MF)
│       └── doc.md                   # CF model documentation
├── main.py                          # End-to-end execution and hybrid evaluation
├── .gitignore
└── README.md
```

---

## 6. Getting Started

### 1) Preprocess Data
Run the preprocessing & EDA pipeline (see `data/doc.md` for details):
```bash
# 100k (Prototyping)
python data/preprocess_100k.py

# 1M (Scaling)
python data/1m/download.py
python data/preprocess_1m.py
```

### 2) Run Hybrid Recommender
Execute the end-to-end pipeline:
```bash
python main.py
```
