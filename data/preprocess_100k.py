"""
MovieLens 100k 데이터 전처리 및 EDA 파이프라인
- 역할: 원본 데이터 정제, 탐색적 데이터 분석(EDA), ID 재색인, Train/Val/Test 분할
- 모델링(TF-IDF, 유사도 계산, SVD 등)에 필요한 피처 엔지니어링은 각 모델 모듈에서 수행합니다.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
    "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"
]


def load_data(data_dir):
    """u.data, u.item, u.user 파일을 읽어 데이터프레임으로 변환."""
    ratings = pd.read_csv(
        os.path.join(data_dir, "u.data"), sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    item_cols = ["item_id", "title", "release_date", "video_release", "imdb_url"] + GENRES
    items = pd.read_csv(
        os.path.join(data_dir, "u.item"), sep="|", names=item_cols,
        encoding="latin-1"
    ).drop(columns=["video_release", "imdb_url"])

    users = pd.read_csv(
        os.path.join(data_dir, "u.user"), sep="|",
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    return ratings, items, users


def clean_data(ratings, items):
    """중복 제거 및 메타데이터에 존재하는 유효한 영화 평점만 필터링."""
    ratings = (ratings.sort_values(["timestamp", "user_id", "item_id"])
                      .drop_duplicates(["user_id", "item_id"], keep="last")
                      .reset_index(drop=True))
    ratings = ratings[ratings.item_id.isin(items.item_id)].reset_index(drop=True)
    return ratings


def run_eda(ratings, items, plot_dir):
    """탐색적 데이터 분석(EDA) 플롯(2x2) 생성 및 저장."""
    os.makedirs(plot_dir, exist_ok=True)
    u_cnt = ratings.groupby("user_id").size()
    i_cnt = ratings.groupby("item_id").size()

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


def reindex_ids(ratings, items, users):
    """user_id, item_id를 0-based 연속 인덱스(u, i)로 매핑."""
    items = items.sort_values("item_id").reset_index(drop=True)
    uid_map = {u: idx for idx, u in enumerate(sorted(ratings.user_id.unique()))}
    iid_map = {m: idx for idx, m in enumerate(items.item_id)}

    ratings = ratings.copy()
    ratings["u"] = ratings.user_id.map(uid_map)
    ratings["i"] = ratings.item_id.map(iid_map)
    items["i"] = items.item_id.map(iid_map)
    users["u"] = users.user_id.map(uid_map)

    # 영화 개봉 연도 및 연대(decade) 파싱 (Content-Based 모델 편의용)
    year_from_title = items.title.str.extract(r"\((\d{4})\)\s*$")[0]
    year_from_date = pd.to_datetime(items.release_date, format="%d-%b-%Y", errors="coerce").dt.year
    items["year"] = pd.to_numeric(year_from_title).fillna(year_from_date)
    items["decade"] = items.year.apply(lambda y: f"{int(y)//10*10}s" if pd.notna(y) else "unknown")

    return ratings, items, users, uid_map, iid_map


def split_train_val_test(ratings, val_ratio=0.1, test_ratio=0.2, seed=SEED):
    """유저별 평점을 7:1:2 비율로 층화 분할."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for _, group in ratings.groupby("u"):
        idx = rng.permutation(len(group))
        n = len(group)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        shuffled = group.iloc[idx]
        splits["test"].append(shuffled.iloc[:n_test])
        splits["val"].append(shuffled.iloc[n_test:n_test + n_val])
        splits["train"].append(shuffled.iloc[n_test + n_val:])

    return {k: pd.concat(v).reset_index(drop=True) for k, v in splits.items()}


def compute_train_stats(train, n_users, n_items):
    """데이터 누수 방지용 Train 세트 기준 기본 통계치."""
    gm = train.rating.mean()
    u_mean = train.groupby("u").rating.mean().reindex(range(n_users)).fillna(gm).values
    i_mean = train.groupby("i").rating.mean().reindex(range(n_items)).fillna(gm).values
    u_cnt = train.groupby("u").size().reindex(range(n_users)).fillna(0).astype(int).values
    return gm, u_mean, i_mean, u_cnt


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 로드 및 정제
    ratings, items, users = load_data(args.data_dir)
    ratings = clean_data(ratings, items)

    # 2. EDA 플롯 생성
    run_eda(ratings, items, args.plot_dir)

    # 3. ID 재색인
    ratings, items, users, uid_map, iid_map = reindex_ids(ratings, items, users)
    n_users, n_items = len(uid_map), len(iid_map)

    # 4. Train / Val / Test (7:1:2) 분할
    splits = split_train_val_test(ratings)

    # 5. 분할 데이터 저장
    keep_cols = ["user_id", "item_id", "u", "i", "rating", "timestamp"]
    for split_name, df in splits.items():
        df[keep_cols].to_csv(os.path.join(args.out_dir, f"{split_name}.csv"), index=False)

    # 6. 메타데이터 및 매핑 저장
    items[["i", "item_id", "title", "release_date", "year", "decade"] + GENRES].to_csv(
        os.path.join(args.out_dir, "items.csv"), index=False
    )
    users[["u", "user_id", "age", "gender", "occupation", "zip"]].to_csv(
        os.path.join(args.out_dir, "users.csv"), index=False
    )
    with open(os.path.join(args.out_dir, "id_maps.json"), "w") as f:
        json.dump({
            "user": {str(k): int(v) for k, v in uid_map.items()},
            "item": {str(k): int(v) for k, v in iid_map.items()}
        }, f)

    # 7. 기본 통계치 저장 (하이브리드 조립 시 N값 참조용)
    gm, u_mean, i_mean, u_cnt = compute_train_stats(splits["train"], n_users, n_items)
    np.savez(
        os.path.join(args.out_dir, "train_stats.npz"),
        global_mean=gm, user_mean=u_mean, item_mean=i_mean, user_count=u_cnt
    )

    print(f"[완료] 전처리 완료 -> {args.out_dir}")
    print(f"       Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/100k/raw")
    parser.add_argument("--out_dir", default="data/100k/preprocessed")
    parser.add_argument("--plot_dir", default="data/plots")
    main(parser.parse_args())
