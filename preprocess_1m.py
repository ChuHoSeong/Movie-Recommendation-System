"""
ML Term Project (Team 5) - Step 1: Data Preprocessing & EDA for MovieLens 1M
=============================================================================
목표: MovieLens 1M 대규모 평점 및 영화 메타데이터 정제, 통계 기반 EDA, 
      누수 방지 Train/Val/Test 분할, 그리고 하이브리드 추천 모델을 위한 
      TF-IDF/희소행렬(CSR) 산출물 생성.

출처 (References):
  - Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
             History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4.
             https://grouplens.org/datasets/movielens/1m/
  - Scipy Sparse: https://docs.scipy.org/doc/scipy/reference/sparse.html
  - Scikit-learn Feature Extraction & Metrics:
    https://scikit-learn.org/stable/modules/feature_extraction.html#tfidfvectorizer

[수업 외 활용 함수 및 라이브러리 설명]
  1. pd.read_csv(..., sep="::", engine="python"):
     - C 파서 엔진은 1글자 구분자만 지원하므로, MovieLens 1M의 2글자 구분자 '::'를
       정규식 파싱하기 위해 Python 엔진을 지정함.
  2. scipy.sparse.csr_matrix(...) :
     - (User × Item) 평점 행렬은 95% 이상이 비어있는 희소행렬(Sparse Matrix)임.
     - 0을 제외한 유효 평점의 (data, (row, col))만 행(Row) 압축 형식으로 저장하여 메모리를 수십 배 절약함.
  3. scipy.sparse.save_npz / load_npz:
     - 희소행렬의 구조를 유지한 채 압축 이진 포맷(.npz)으로 초고속 저장/로드 수행.
  4. sklearn.feature_extraction.text.TfidfVectorizer:
     - 영화 장르/연대/제목 단어 토큰의 빈도(TF)와 역문서빈도(IDF)를 곱해 단어별 중요도 가중 벡터를 생성함.
     - sublinear_tf=True: TF 값에 1 + log(tf)를 취해 특정 단어 빈도의 과도한 영향 완화.
  5. sklearn.metrics.pairwise.cosine_similarity:
     - TF-IDF로 임베딩된 영화 벡터 간의 코사인 각도를 내적하여 [-1, 1] 범위의 영화 간 유사도 행렬 산출.
  6. numpy.random.default_rng(seed):
     - Python 기본 random 모듈보다 빠르고 재현 가능한 고성능 Mersenne Twister/PCG64 난수 생성기.
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

# 1M 장르 매핑 (100k와 동일한 19개 장르 컬럼 체계 유지)
GENRE_MAP = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Children's": "Childrens",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Drama": "Drama",
    "Fantasy": "Fantasy",
    "Film-Noir": "FilmNoir",
    "Horror": "Horror",
    "Musical": "Musical",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Sci-Fi": "SciFi",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western"
}
# 100k와의 규격 일치를 위해 'unknown'을 첫 번째 장르로 포함 (총 19개)
GENRES = ["unknown"] + list(GENRE_MAP.values())

# 제목 불용어 (TF-IDF 노이즈 방지용)
STOP_TITLE = {"the", "a", "an", "of", "and", "in", "to", "for", "on", "at", "la", "le", "de"}


def load_data(data_dir):
    """
    ratings.dat, movies.dat, users.dat 파일을 읽어 데이터프레임으로 변환.
    MovieLens 1M은 '::' 구분자와 latin-1 인코딩을 사용함.
    """
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"), sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    movies_raw = pd.read_csv(
        os.path.join(data_dir, "movies.dat"), sep="::", engine="python",
        names=["item_id", "title", "genres_str"], encoding="latin-1"
    )

    items = movies_raw.copy()
    for g in GENRES:
        items[g] = 0

    # 파이프(|)로 구분된 장르 문자열을 0/1 바이너리 컬럼으로 전개
    for idx, row in items.iterrows():
        if pd.notna(row["genres_str"]):
            for g_raw in row["genres_str"].split("|"):
                g_clean = GENRE_MAP.get(g_raw)
                if g_clean:
                    items.at[idx, g_clean] = 1

    # 100k와 완벽한 호환을 위해 장르가 없는 경우 unknown = 1 부여
    items["unknown"] = (items[list(GENRE_MAP.values())].sum(axis=1) == 0).astype(int)

    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"), sep="::", engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"]
    )
    return ratings, items, users


def quality_check(ratings, items):
    """
    데이터 품질 진단 및 정제:
      - 결측치 및 중복 평점 점검
      - 복합키(timestamp, user_id, item_id) 정렬을 통한 결정론적(Deterministic) 정렬 보장
      - 메타데이터에 없는 영화 평점 필터링
    """
    report = {}
    report["n_ratings_raw"] = len(ratings)
    report["ratings_missing"] = int(ratings.isna().sum().sum())
    report["duplicate_user_item"] = int(ratings.duplicated(["user_id", "item_id"]).sum())
    report["rating_values"] = sorted(ratings["rating"].unique().tolist())

    # 동일 타임스탬프 동순위로 인한 환경별 셔플 불일치를 방지하기 위한 복합 정렬
    ratings = (ratings.sort_values(["timestamp", "user_id", "item_id"])
                      .drop_duplicates(["user_id", "item_id"], keep="last")
                      .sort_values(["user_id", "item_id"]))

    r_items, m_items = set(ratings.item_id), set(items.item_id)
    report["items_in_ratings_not_in_meta"] = len(r_items - m_items)
    report["items_in_meta_never_rated"] = len(m_items - r_items)
    ratings = ratings[ratings.item_id.isin(m_items)]

    no_genre = items[items[GENRES].sum(axis=1) == 0]
    report["items_without_genre"] = no_genre.item_id.tolist()
    report["n_ratings_clean"] = len(ratings)
    return ratings.reset_index(drop=True), report


def run_eda(ratings, items, plot_dir):
    """
    탐색적 데이터 분석(EDA):
      - 희소도(Sparsity), 평점 평균, 유저/아이템별 롱테일 분포 통계 산출
      - 2x2 서브플롯 시각화 차트 생성 및 저장
    """
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
    ax[0, 0].set(title="Rating distribution (1M)", xlabel="rating", ylabel="count")
    ax[0, 1].hist(u_cnt, bins=50, color="#55A868")
    ax[0, 1].set(title="Ratings per user (1M long tail)", xlabel="# ratings", ylabel="# users")
    ax[1, 0].hist(i_cnt, bins=50, color="#C44E52")
    ax[1, 0].set(title="Ratings per item (1M long tail)", xlabel="# ratings", ylabel="# items")
    ax[1, 1].barh(GENRES, items[GENRES].sum().values, color="#8172B2")
    ax[1, 1].set(title="Movies per genre (1M)", xlabel="# movies")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "eda_overview_1m.png"), dpi=130)
    plt.close()
    return stats


def reindex(ratings, items):
    """
    원본 user_id, item_id를 0..N-1의 연속된 정수 인덱스(u, i)로 매핑.
    희소행렬 인덱싱 및 모델 잠재요인 임베딩 최적화 목적.
    """
    items = items.sort_values("item_id").reset_index(drop=True)
    uid_map = {u: i for i, u in enumerate(sorted(ratings.user_id.unique()))}
    iid_map = {m: i for i, m in enumerate(items.item_id)}
    ratings = ratings.copy()
    ratings["u"] = ratings.user_id.map(uid_map)
    ratings["i"] = ratings.item_id.map(iid_map)
    items["i"] = items.item_id.map(iid_map)
    return ratings, items, uid_map, iid_map


def build_content_features(items):
    """
    콘텐츠 기반 추천을 위한 피처 엔지니어링:
      - 제목에서 연도 추출 -> decade 토큰 생성
      - 장르(3배 가중치) + decade + 제목 키워드 결합 -> TF-IDF 문서 생성
      - TfidfVectorizer 및 코사인 유사도 행렬 사전 계산
    """
    items = items.copy()
    year_from_title = items.title.str.extract(r"\((\d{4})\)\s*$")[0]
    items["year"] = pd.to_numeric(year_from_title)
    items["clean_title"] = items.title.str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    items["decade"] = items.year.apply(lambda y: f"decade{int(y)//10*10}" if pd.notna(y) else "decadeunk")

    def doc(row):
        genre_tokens = [g.lower() for g in GENRES if row[g] == 1]
        words = [f"w_{w}" for w in re.findall(r"[a-z]+", row.clean_title.lower())
                 if w not in STOP_TITLE and len(w) > 2]
        # 장르 중요도를 반영하기 위해 장르 토큰 3회 반복
        return " ".join(genre_tokens * 3 + [row.decade] + words)

    items["content_doc"] = items.apply(doc, axis=1)
    vec = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
    tfidf = vec.fit_transform(items.content_doc)
    sim = cosine_similarity(tfidf)
    return items, tfidf, sim, vec


def split_per_user(ratings, val_frac=0.1, test_frac=0.2, seed=SEED):
    """
    유저별 층화 분할(User-stratified Split 7 : 1 : 2):
      - 모든 유저가 Train, Val, Test 세트에 고루 포함되도록 유저별 분할
      - 고정 시드 난수 순열(Permutation)을 적용하여 엄격한 재현성 보장
    """
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
    """
    콜드스타트 시뮬레이션 데이터셋 생성 (Given-k):
      - Train 세트에서 각 유저당 평점을 최대 k개만 남겨 극단적 라이트 유저 환경 모의.
    """
    shuffled = train.sample(frac=1.0, random_state=seed)
    return shuffled.groupby("u").head(k).reset_index(drop=True)


def to_csr(df, n_users, n_items):
    """DataFrame 평점 데이터를 Scipy 행 압축 희소행렬(CSR Matrix)로 변환."""
    return sparse.csr_matrix((df.rating.values.astype(np.float32), (df.u.values, df.i.values)),
                             shape=(n_users, n_items))


def train_stats(train, n_users, n_items):
    """
    데이터 누수(Data Leakage) 차단을 위한 통계치 산출:
      - 반드시 Train 세트만을 사용하여 글로벌 평균, 유저/아이템별 평균, 평가수(N) 계산.
    """
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

    # 데이터셋 저장
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
    p.add_argument("--data_dir", default="ml-1m")
    p.add_argument("--out_dir", default="data/preprocessed_1m")
    p.add_argument("--plot_dir", default="plots")
    main(p.parse_args())
