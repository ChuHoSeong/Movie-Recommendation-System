import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def find_root():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data" / "100k" / "preprocessed").exists():
            return parent
    raise FileNotFoundError("data/100k/preprocessed 폴더를 찾을 수 없습니다.")


ROOT = find_root()
DATA_DIR = ROOT / "data" / "100k" / "preprocessed"
ITEM_FILE = DATA_DIR / "items.csv"
PLOT_DIR = Path(__file__).resolve().parent / "plots"

GENRES = ["unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy",
          "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror",
          "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"]

EXPERIMENTS = [
    {"name": "Genre k20", "features": ("genre",), "k": 20, "damp": 0.5},
    {"name": "Genre+Title k20", "features": ("genre", "title"), "k": 20, "damp": 0.5},
    {"name": "Genre+Decade k20", "features": ("genre", "decade"), "k": 20, "damp": 0.5},
    {"name": "G+T+D k10", "features": ("genre", "title", "decade"), "k": 10, "damp": 0.5},
    {"name": "G+T+D k20", "features": ("genre", "title", "decade"), "k": 20, "damp": 0.5},
    {"name": "G+T+D k50", "features": ("genre", "title", "decade"), "k": 50, "damp": 0.5},
    {"name": "G+T+D k20 damp2", "features": ("genre", "title", "decade"), "k": 20, "damp": 2.0},
]


def load_split(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path, usecols=["u", "i", "rating"])
    if df.empty or df[["u", "i", "rating"]].isna().any().any():
        raise ValueError(f"Empty or invalid data: {path}")
    return df


def build_documents(items, features):
    docs = []
    for _, row in items.iterrows():
        tokens = []
        if "genre" in features:
            tokens += [f"g_{g}" for g in GENRES if row[g] == 1]
        if "title" in features:
            title = re.sub(r"\(\d{4}\)", " ", str(row["title"]))
            tokens += [f"t_{w}" for w in re.findall(r"[a-z0-9]+", title.lower())
                       if len(w) > 2]
        if "decade" in features and str(row["decade"]) != "unknown":
            tokens.append(f"d_{row['decade']}")
        docs.append(" ".join(tokens) if tokens else "empty")
    return docs


def evaluate(model, df):
    actual = df["rating"].to_numpy(dtype=np.float64)
    predicted = model.predict(df["u"].to_numpy(), df["i"].to_numpy())
    errors = actual - predicted
    return float(np.sqrt(np.mean(errors ** 2))), float(np.mean(np.abs(errors)))


class TfidfContentBased:
    def __init__(self, n_users, items, features=("genre", "title", "decade"),
                 k=20, damp=0.5, mean_shrink=5.0):
        self.n_users = int(n_users)
        self.k = int(k)
        self.damp = float(damp)
        self.mean_shrink = float(mean_shrink)
        items = items.sort_values("i").reset_index(drop=True)
        self.n_items = len(items)

        tfidf = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True)
        matrix = tfidf.fit_transform(build_documents(items, features))
        self.sim = (matrix @ matrix.T).toarray()
        np.fill_diagonal(self.sim, 0.0)
        self.global_mean = None

    def fit(self, train):
        users = train["u"].to_numpy(dtype=np.int64)
        items = train["i"].to_numpy(dtype=np.int64)
        ratings = train["rating"].to_numpy(dtype=np.float64)

        self.global_mean = float(ratings.mean())
        count = np.bincount(users, minlength=self.n_users)
        total = np.bincount(users, weights=ratings, minlength=self.n_users)
        self.user_mean = (total + self.mean_shrink * self.global_mean) / (
            count + self.mean_shrink
        )
        self.user_items = [[] for _ in range(self.n_users)]
        self.user_resid = [[] for _ in range(self.n_users)]
        for u, i, r in zip(users, items, ratings):
            self.user_items[u].append(i)
            self.user_resid[u].append(r - self.user_mean[u])
        self.user_items = [np.array(x, dtype=np.int64) for x in self.user_items]
        self.user_resid = [np.array(x, dtype=np.float64) for x in self.user_resid]
        return self

    def predict(self, users, items):
        if self.global_mean is None:
            raise RuntimeError("Call fit() before predict().")
        users_array, items_array = np.broadcast_arrays(
            np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)
        )
        shape = users_array.shape
        u_all = users_array.ravel()
        i_all = items_array.ravel()
        predicted = np.full(u_all.size, self.global_mean, dtype=np.float64)

        valid_u = (u_all >= 0) & (u_all < self.n_users)
        predicted[valid_u] = self.user_mean[u_all[valid_u]]

        for u in np.unique(u_all[valid_u]):
            rated = self.user_items[u]
            if rated.size == 0:
                continue
            pos = np.where((u_all == u) & (i_all >= 0) & (i_all < self.n_items))[0]
            if pos.size == 0:
                continue
            sims = self.sim[np.ix_(i_all[pos], rated)]
            k = min(self.k, rated.size)
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            top_sims = np.take_along_axis(sims, top, axis=1)
            top_resid = self.user_resid[u][top]
            num = (top_sims * top_resid).sum(axis=1)
            den = np.abs(top_sims).sum(axis=1) + self.damp
            predicted[pos] += num / den

        predicted = np.clip(predicted, 1.0, 5.0).reshape(shape)
        return float(predicted) if predicted.ndim == 0 else predicted


def plot_experiments(results):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([r["name"] for r in results], [r["val_rmse"] for r in results])
    ax.set(ylabel="Validation RMSE", title="TF-IDF content-based comparison (lower is better)")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "cb_tfidf_validation_comparison.png", dpi=150)
    plt.close(fig)


def main():
    train = load_split("train.csv")
    val = load_split("val.csv")
    test = load_split("test.csv")
    items = pd.read_csv(ITEM_FILE)
    n_users = int(train["u"].max()) + 1
    print(f"Data: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")

    results, best_model, best_score = [], None, float("inf")
    for params in EXPERIMENTS:
        model = TfidfContentBased(
            n_users, items, features=params["features"],
            k=params["k"], damp=params["damp"],
        ).fit(train)
        val_rmse, val_mae = evaluate(model, val)
        results.append({**params, "val_rmse": val_rmse, "val_mae": val_mae})
        print(f"{params['name']:<18} RMSE={val_rmse:.5f} MAE={val_mae:.5f}")
        if val_rmse < best_score:
            best_score, best_model = val_rmse, model

    plot_experiments(results)
    test_rmse, test_mae = evaluate(best_model, test)
    winner = next(r for r in results if r["val_rmse"] == best_score)
    print(f"\nBest model: {winner['name']} (features={winner['features']}, "
          f"k={winner['k']}, damp={winner['damp']})")
    print(f"Final TEST: RMSE={test_rmse:.5f}, MAE={test_mae:.5f}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()