"""
ML Term Project (Team 5) - Step 1: Data Preprocessing & EDA for MovieLens 100k
==============================================================================
목표: MovieLens 100k 평점 및 영화 메타데이터 정제, 통계 기반 EDA, 
      누수 방지 Train/Val/Test 유저별 층화 분할, 그리고 하이브리드 추천 모델을 위한 
      TF-IDF/희소행렬(CSR Matrix) 산출물 생성.

출처 (References):
  - Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
             History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4.
             https://grouplens.org/datasets/movielens/100k/
  - Scipy Sparse Matrix: https://docs.scipy.org/doc/scipy/reference/sparse.html
  - Scikit-learn TF-IDF & Cosine Similarity:
    https://scikit-learn.org/stable/modules/feature_extraction.html#tfidfvectorizer

[수업 외 활용 함수 및 라이브러리 설명]
  1. scipy.sparse.csr_matrix(...) :
     - 90% 이상이 0으로 채워진 (User × Item) 평점 행렬을 압축하기 위해 행 압축(Compressed Sparse Row) 사용.
     - parameters: (data, (row_idx, col_idx)), shape
  2. scipy.sparse.save_npz / load_npz:
     - 희소행렬의 구조(인덱스, 데이터 포인터)를 유지한 채 압축 이진 포맷(.npz)으로 초고속 저장/로드 수행.
  3. sklearn.feature_extraction.text.TfidfVectorizer:
     - 영화 장르/연대/제목 토큰의 단어 빈도(TF)와 역문서빈도(IDF)를 결합하여 중요도 가중 임베딩 벡터 생성.
     - parameters: token_pattern(토큰 추출 정규식), sublinear_tf(log 스케일링으로 이상치 빈도 완화).
  4. sklearn.metrics.pairwise.cosine_similarity:
     - TF-IDF로 벡터화된 영화 간 코사인 유사도를 계산하여 아이템 간 유사도 행렬을 생성함.
  5. numpy.random.default_rng(seed):
     - Python 기본 난수 생성기보다 빠르고 재현 가능한 PCG64 기반 난수 생성기.
"""

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")  # GUI 윈도우 없이 백그라운드 파일 저장 모드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42

# MovieLens 100k 19개 장르 (u.genre 순서와 일치)
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"
]

# 제목 불용어 (TF-IDF 노이즈 방지용)
STOP_TITLE = {"the", "a", "an", "of", "and", "in", "to", "for", "on", "at", "la", "le", "de"}


def load_data(data_dir):
    """u.data, u.item, u.user 파일을 읽어 데이터프레임으로 변환."""
    ratings = pd.read_csv(
        os.path.join(data_dir, "u.data"), sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    item_cols = ["item_id", "title", "release_date", "video_release", "imdb_url"] + GENRES
    items = pd.read_csv(
        os.path.join(data_dir, "u.item"), sep="|", names=item_cols,
        encoding="latin-1"  # u.item은 latin-1 인코딩
    )
    items = items.drop(columns=["video_release", "imdb_url"])  # 결측률 100% 컬럼 제거

    users = pd.read_csv(
        os.path.join(data_dir, "u.user"), sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    return ratings, items, users


def quality_check(ratings, items):
    """
    데이터 품질 점검 및 정제:
      - 동일 타임스탬프 동순위로 인한 환경별 셔플 불일치를 방지하기 위해 복합키 정렬 적용.
      - 메타데이터에 없는 영화 평점 필터링.
    """
    report = {}
    report["n_ratings_raw"] = len(ratings)
    report["ratings_missing"] = int(ratings.isna().sum().sum())
    report["duplicate_user_item"] = int(ratings.duplicated(["user_id", "item_id"]).sum())
    report["rating_values"] = sorted(ratings["rating"].unique().tolist())

    # 동일 타임스탬프 동순위로 인한 환경별 분할 불일치 방지 (결정론적 정렬)
    ratings = (ratings.sort_values(["timestamp", "user_id", "item_id"])
                      .drop_duplicates(["user_id", "item_id"], keep="last")
                      .sort_values(["user_id", "item_id"]))

    r_items, m_items = set(ratings.item_id), set(items.item_id)
    report["items_in_ratings_not_in_meta"] = len(r_items - m_items)
    report["items_in_meta_never_rated"] = len(m_items - r_items)
    ratings = ratings[ratings.item_id.isin(m_items)]

    no_genre = items[items[GENRES].sum(axis=1) == 0]
    report["items_without_genre"] = no_genre.item_id.tolist()
    report["items_missing_release_date"] = items[items.release_date.isna()].item_id.tolist()
    report["n_ratings_clean"] = len(ratings)
    return ratings.reset_index(drop=True), report


def run_eda(ratings, items, plot_dir):
    """탐색적 데이터 분석(EDA) 통계 산출 및 2x2 분포 플롯 저장."""
    os.makedirs(plot_dir, exist_ok=True)
    n_users, n_items = ratings.user_id.nunique(), ratings.item_id.nunique()
    stats = {
        "n_users": n_users, "n_items": n_items, "n_ratings": len(ratings),
        "sparsity_pct": round(100 * (1 - len(ratings) / (n_users * n_items)), 2),
        "rating_mean": round(ratings.rating.mean(), 3),
    }
    u_cnt = ratings.groupby("user_id").size()
    i_cnt = ratings.groupby("item_id").size()
    stats.update({
        "user_count_min": int(u_cnt.min()), "user_count_median": float(u_cnt.median()),
        "user_count_max": int(u_cnt.max()),
        "item_count_min": int(i_cnt.min()), "item_count_median": float(i_cnt.median()),
        "item_count_max": int(i_cnt.max()),
        "items_with_le5_ratings_pct": round(100 * (i_cnt <= 5).mean(), 1),
    })

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ratings.rating.value_counts().sort_index().plot.bar(ax=ax[0, 0], color="#4C72B0")
    ax[0, 0].set(title="Rating distribution", xlabel="rating", ylabel="count")
    ax[0, 1].hist(u_cnt, bins=50, color="#55A868")
    ax[0, 1].set(title="Ratings per user (long tail)", xlabel="# ratings", ylabel="# users")
    ax[1, 0].hist(i_cnt, bins=50, color="#C44E52")
    ax[1, 0].set(title="Ratings per item (long tail)", xlabel="# ratings", ylabel="# items")
    ax[1, 1].barh(GENRES, items[GENRES].sum().values, color="#8172B2")
    ax[1, 1].set(title="Movies per genre", xlabel="# movies")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "eda_overview.png"), dpi=130)
    plt.close()
    return stats


def reindex(ratings, items):
    """원본 user_id/item_id -> 0..n-1 연속 인덱스 매핑."""
    items = items.sort_values("item_id").reset_index(drop=True)
    uid_map = {u: i for i, u in enumerate(sorted(ratings.user_id.unique()))}
    iid_map = {m: i for i, m in enumerate(items.item_id)}
    ratings = ratings.copy()
    ratings["u"] = ratings.user_id.map(uid_map)
    ratings["i"] = ratings.item_id.map(iid_map)
    items["i"] = items.item_id.map(iid_map)
    return ratings, items, uid_map, iid_map


def build_content_features(items):
    """장르(3배 가중치), 개봉연대, 제목 키워드를 결합한 TF-IDF 및 코사인 유사도 행렬 생성."""
    items = items.copy()
    year_from_title = items.title.str.extract(r"\((\d{4})\)\s*$")[0]
    year_from_date = pd.to_datetime(items.release_date, format="%d-%b-%Y", errors="coerce").dt.year
    items["year"] = pd.to_numeric(year_from_title).fillna(year_from_date)
    items["clean_title"] = items.title.str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    items["decade"] = items.year.apply(lambda y: f"decade{int(y)//10*10}" if pd.notna(y) else "decadeunk")

    def doc(row):
        genre_tokens = [g.lower() for g in GENRES if row[g] == 1]
        words = [f"w_{w}" for w in re.findall(r"[a-z]+", row.clean_title.lower())
                 if w not in STOP_TITLE and len(w) > 2]
        return " ".join(genre_tokens * 3 + [row.decade] + words)

    items["content_doc"] = items.apply(doc, axis=1)
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    tfidf = vec.fit_transform(items.content_doc)
    sim = cosine_similarity(tfidf)
    return items, tfidf, sim, vec


def split_per_user(ratings, val_frac=0.1, test_frac=0.2, seed=SEED):
    """유저별 평점을 7:1:2 비율로 층화 분할(User-stratified)."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, g in ratings.groupby("u"):
        idx = rng.permutation(len(g))
        n = len(g)
        n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
        g = g.iloc[idx]
        parts["test"].append(g.iloc[:n_test])
        parts["val"].append(g.iloc[n_test:n_test + n_val])
        parts["train"].append(g.iloc[n_test + n_val:])
    return {k: pd.concat(v).reset_index(drop=True) for k, v in parts.items()}


def make_given_k(train, k, seed=SEED):
    """콜드스타트 시뮬레이션: 유저당 k개의 평가만 남김."""
    shuffled = train.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("u").head(k).reset_index(drop=True)


def to_csr(df, n_users, n_items):
    """Scipy Compressed Sparse Row (CSR) 희소행렬 변환."""
    return sparse.csr_matrix((df.rating.values.astype(np.float32), (df.u.values, df.i.values)),
                             shape=(n_users, n_items))


def train_stats(train, n_users, n_items):
    """데이터 누수 방지를 위한 Train 세트 기준 통계치(글로벌/유저/아이템 평균, 유저 평가수 N) 산출."""
    gm = train.rating.mean()
    u_mean = train.groupby("u").rating.mean().reindex(range(n_users)).fillna(gm)
    i_mean = train.groupby("i").rating.mean().reindex(range(n_items)).fillna(gm)
    u_cnt = train.groupby("u").size().reindex(range(n_users)).fillna(0).astype(int)
    return gm, u_mean.values, i_mean.values, u_cnt.values


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    ratings, items, users = load_data(args.data_dir)
    ratings, qc = quality_check(ratings, items)
    eda = run_eda(ratings, items, args.plot_dir)
    ratings, items, uid_map, iid_map = reindex(ratings, items)
    n_users, n_items = len(uid_map), len(iid_map)

    items, tfidf, sim, vec = build_content_features(items)
    splits = split_per_user(ratings)

    train_items = set(splits["train"].i)
    split_info = {}
    for name in ("val", "test"):
        df = splits[name]
        split_info[f"{name}_size"] = len(df)
        split_info[f"{name}_items_unseen_in_train_pct"] = round(100 * (~df.i.isin(train_items)).mean(), 2)
    split_info["train_size"] = len(splits["train"])
    split_info["train_user_count_min"] = int(splits["train"].groupby("u").size().min())

    # 데이터 분할 파일 저장
    keep = ["user_id", "item_id", "u", "i", "rating", "timestamp"]
    for name, df in splits.items():
        df[keep].to_csv(os.path.join(args.out_dir, f"{name}.csv"), index=False)
    for k in (3, 5, 10, 20):
        make_given_k(splits["train"], k)[keep].to_csv(
            os.path.join(args.out_dir, f"train_given{k}.csv"), index=False)

    items[["i", "item_id", "title", "clean_title", "year", "decade"] + GENRES + ["content_doc"]] \
        .to_csv(os.path.join(args.out_dir, "items.csv"), index=False)
    users.assign(u=users.user_id.map(uid_map)).to_csv(os.path.join(args.out_dir, "users.csv"), index=False)
    sparse.save_npz(os.path.join(args.out_dir, "item_tfidf.npz"), tfidf)
    np.save(os.path.join(args.out_dir, "item_item_cosine.npy"), sim.astype(np.float32))
    sparse.save_npz(os.path.join(args.out_dir, "R_train.npz"), to_csr(splits["train"], n_users, n_items))

    gm, u_mean, i_mean, u_cnt = train_stats(splits["train"], n_users, n_items)
    np.savez(os.path.join(args.out_dir, "train_stats.npz"),
             global_mean=gm, user_mean=u_mean, item_mean=i_mean, user_count=u_cnt)
    with open(os.path.join(args.out_dir, "id_maps.json"), "w") as f:
        json.dump({"user": {str(k): v for k, v in uid_map.items()},
                   "item": {str(k): v for k, v in iid_map.items()}}, f)

    summary = {"quality": qc, "eda": eda, "split": split_info,
               "tfidf_shape": list(tfidf.shape), "train_global_mean": float(gm)}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    default_data = "data/raw" if os.path.exists("data/raw") else "data/ml-100k"
    default_out = "data/preprocessed" if os.path.exists("data/raw") else "processed"
    p.add_argument("--data_dir", default=default_data)
    p.add_argument("--out_dir", default=default_out)
    p.add_argument("--plot_dir", default="plots")
    main(p.parse_args())
