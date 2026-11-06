"""MovieLens Model-based CF: biased matrix factorization (Funk SVD)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "100k" / "preprocessed"
PLOT_DIR = Path(__file__).resolve().parent / "plots"
SEED = 42

# All candidates use the same maximum epoch budget and early stopping patience.
# Candidates are compared on validation data; only the selected model is tested.
EXPERIMENTS = [
    {"name": "Bias only", "n_factors": 0, "n_epochs": 50, "lr": 0.005, "reg": 0.02},
    {"name": "MF-20", "n_factors": 20, "n_epochs": 50, "lr": 0.005, "reg": 0.02},
    {"name": "MF-50", "n_factors": 50, "n_epochs": 50, "lr": 0.005, "reg": 0.02},
    {"name": "MF-50-reg", "n_factors": 50, "n_epochs": 50, "lr": 0.005, "reg": 0.05},
    {"name": "MF-50-lr", "n_factors": 50, "n_epochs": 50, "lr": 0.01, "reg": 0.02},
]


def load_split(filename):
    path = DATA_DIR / filename
    df = pd.read_csv(path, usecols=["u", "i", "rating"])
    if df.empty or df[["u", "i", "rating"]].isna().any().any():
        raise ValueError(f"Empty or invalid data: {path}")
    return df


def evaluate(model, df):
    actual = df["rating"].to_numpy(dtype=np.float64)
    predicted = model.predict(df["u"].to_numpy(), df["i"].to_numpy())
    errors = actual - predicted
    return float(np.sqrt(np.mean(errors ** 2))), float(np.mean(np.abs(errors)))


class BiasedMF:
    """Funk SVD: global mean + user/item biases + user/item latent factors."""

    def __init__(self, n_users, n_items, n_factors=50, n_epochs=20,
                 lr=0.005, reg=0.02, seed=SEED, patience=6):
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_factors = int(n_factors)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.reg = float(reg)
        self.seed = int(seed)
        self.patience = int(patience)
        self.history = []
        self.best_epoch = 0
        self.global_mean = None

    def fit(self, train, val):
        rng = np.random.default_rng(self.seed)
        users = train["u"].to_numpy(dtype=np.int64)
        items = train["i"].to_numpy(dtype=np.int64)
        ratings = train["rating"].to_numpy(dtype=np.float64)

        self.global_mean = float(np.mean(ratings))
        self.user_bias = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias = np.zeros(self.n_items, dtype=np.float64)
        self.user_factors = rng.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.1, (self.n_items, self.n_factors))
        self.known_users = np.bincount(users, minlength=self.n_users) > 0
        self.known_items = np.bincount(items, minlength=self.n_items) > 0

        best_rmse = float("inf")
        best_state = None
        no_improvement = 0

        for epoch in range(1, self.n_epochs + 1):
            for index in rng.permutation(len(ratings)):
                u = users[index]
                i = items[index]
                estimate = self.global_mean + self.user_bias[u] + self.item_bias[i]
                if self.n_factors:
                    estimate += np.dot(self.user_factors[u], self.item_factors[i])
                error = ratings[index] - estimate

                self.user_bias[u] += self.lr * (error - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (error - self.reg * self.item_bias[i])
                if self.n_factors:
                    p = self.user_factors[u]
                    q = self.item_factors[i]
                    old_p = p.copy()
                    p += self.lr * (error * q - self.reg * p)
                    q += self.lr * (error * old_p - self.reg * q)

            train_rmse, _ = evaluate(self, train)
            val_rmse, val_mae = evaluate(self, val)
            self.history.append((epoch, train_rmse, val_rmse, val_mae))

            if val_rmse < best_rmse - 1e-5:
                best_rmse = val_rmse
                self.best_epoch = epoch
                best_state = (
                    self.user_bias.copy(), self.item_bias.copy(),
                    self.user_factors.copy(), self.item_factors.copy(),
                )
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    break

        (self.user_bias, self.item_bias,
         self.user_factors, self.item_factors) = best_state
        return self

    def predict(self, users, items):
        """Predict 1–5 rating(s) for internal 0-based `u`, `i` indices."""
        if self.global_mean is None:
            raise RuntimeError("Call fit() before predict().")
        users_array, items_array = np.broadcast_arrays(
            np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64)
        )
        shape = users_array.shape
        u = users_array.ravel()
        i = items_array.ravel()
        predicted = np.full(u.size, self.global_mean, dtype=np.float64)
        valid_u = (u >= 0) & (u < self.n_users)
        valid_i = (i >= 0) & (i < self.n_items)
        u_known = np.zeros(u.size, dtype=bool)
        i_known = np.zeros(i.size, dtype=bool)
        u_known[valid_u] = self.known_users[u[valid_u]]
        i_known[valid_i] = self.known_items[i[valid_i]]
        predicted[u_known] += self.user_bias[u[u_known]]
        predicted[i_known] += self.item_bias[i[i_known]]
        both = u_known & i_known
        if self.n_factors and np.any(both):
            predicted[both] += np.einsum(
                "ij,ij->i", self.user_factors[u[both]], self.item_factors[i[both]]
            )
        predicted = np.clip(predicted, 1.0, 5.0).reshape(shape)
        return float(predicted) if predicted.ndim == 0 else predicted


def plot_experiments(results, best_model):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    names = [row["name"] for row in results]
    rmse = [row["val_rmse"] for row in results]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, rmse)
    ax.set(ylabel="Validation RMSE", title="Model comparison (lower is better)")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "validation_comparison.png", dpi=150)
    plt.close(fig)

    history = np.asarray(best_model.history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history[:, 0], history[:, 1], label="Train RMSE")
    ax.plot(history[:, 0], history[:, 2], label="Validation RMSE")
    ax.axvline(best_model.best_epoch, linestyle="--", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="RMSE", title="Selected model learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)


def main():
    train = load_split("train.csv")
    val = load_split("val.csv")
    test = load_split("test.csv")
    # Determine model dimensions from training data only.
    n_users = int(train["u"].max()) + 1
    n_items = int(train["i"].max()) + 1
    print(f"Data: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")

    results = []
    best_model = None
    best_score = float("inf")

    for params in EXPERIMENTS:
        model = BiasedMF(
            n_users=n_users, n_items=n_items,
            n_factors=params["n_factors"], n_epochs=params["n_epochs"],
            lr=params["lr"], reg=params["reg"],
        ).fit(train, val)
        val_rmse, val_mae = evaluate(model, val)
        results.append({**params, "val_rmse": val_rmse, "val_mae": val_mae,
                        "best_epoch": model.best_epoch})
        print(f"{params['name']:<12} RMSE={val_rmse:.5f} MAE={val_mae:.5f} "
              f"best_epoch={model.best_epoch} stopped_epoch={len(model.history)}")
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model

    plot_experiments(results, best_model)
    test_rmse, test_mae = evaluate(best_model, test)
    winner = next(row for row in results if row["val_rmse"] == best_score)
    print(f"\nBest model: {winner['name']} "
          f"(factors={winner['n_factors']}, lr={winner['lr']}, "
          f"reg={winner['reg']}, epoch={winner['best_epoch']})")
    print(f"Final TEST: RMSE={test_rmse:.5f}, MAE={test_mae:.5f}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()