"""200 Epoch 강제 완주 학습 곡선 시각화 스크립트 (run_200_epochs.py)
- 조기 종료(Early stopping) 없이 1부터 200 Epoch까지 전부 학습
- Train 오차가 계속 떨어지는 동안 Validation 오차가 반등(U자)하는 과적합(Overfitting) 과정을 명확히 시각화
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_based import BiasedMF, load_split, evaluate, PLOT_DIR, SEED


def main():
    train = load_split("train.csv")
    val = load_split("val.csv")
    test = load_split("test.csv")

    n_users = int(train["u"].max()) + 1
    n_items = int(train["i"].max()) + 1

    print("=== MF-50-reg 모델 200 Epoch 강제 학습 시작 ===")
    print("조기 종료(patience) 없이 200 Epoch 전체를 완주하여 과적합 곡선을 관찰합니다.")

    # patience를 200으로 설정하여 조기 종료 없이 200 Epoch 전체 완주
    model = BiasedMF(
        n_users=n_users,
        n_items=n_items,
        n_factors=50,
        n_epochs=200,
        lr=0.005,
        reg=0.05,
        seed=SEED,
        patience=200,
    )
    
    # 학습 진행 (매 10 Epoch마다 로그 출력)
    rng = np.random.default_rng(model.seed)
    users = train["u"].to_numpy(dtype=np.int64)
    items = train["i"].to_numpy(dtype=np.int64)
    ratings = train["rating"].to_numpy(dtype=np.float64)

    model.global_mean = float(np.mean(ratings))
    model.user_bias = np.zeros(model.n_users, dtype=np.float64)
    model.item_bias = np.zeros(model.n_items, dtype=np.float64)
    model.user_factors = rng.normal(0, 0.1, (model.n_users, model.n_factors))
    model.item_factors = rng.normal(0, 0.1, (model.n_items, model.n_factors))
    model.known_users = np.bincount(users, minlength=model.n_users) > 0
    model.known_items = np.bincount(items, minlength=model.n_items) > 0

    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, 201):
        for index in rng.permutation(len(ratings)):
            u = users[index]
            i = items[index]
            estimate = model.global_mean + model.user_bias[u] + model.item_bias[i]
            if model.n_factors:
                estimate += np.dot(model.user_factors[u], model.item_factors[i])
            error = ratings[index] - estimate

            model.user_bias[u] += model.lr * (error - model.reg * model.user_bias[u])
            model.item_bias[i] += model.lr * (error - model.reg * model.item_bias[i])
            if model.n_factors:
                p = model.user_factors[u]
                q = model.item_factors[i]
                old_p = p.copy()
                p += model.lr * (error * q - model.reg * p)
                q += model.lr * (error * old_p - model.reg * q)

        train_rmse, _ = evaluate(model, train)
        val_rmse, val_mae = evaluate(model, val)
        model.history.append((epoch, train_rmse, val_rmse, val_mae))

        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            model.best_epoch = epoch
            best_state = (
                model.user_bias.copy(), model.item_bias.copy(),
                model.user_factors.copy(), model.item_factors.copy(),
            )

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/200 | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | (Best Epoch: {model.best_epoch})")

    # 그래프 생성
    history = np.asarray(model.history)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(history[:, 0], history[:, 1], label="Train RMSE", color="#1f77b4", linewidth=1.8)
    ax.plot(history[:, 0], history[:, 2], label="Validation RMSE", color="#ff7f0e", linewidth=1.8)
    ax.axvline(model.best_epoch, linestyle="--", color="#2ca02c", label=f"Best epoch ({model.best_epoch})")
    
    ax.set_title("Biased MF Learning Curve (1 ~ 200 Epochs Overfitting Observation)", fontsize=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10.5)
    
    fig.tight_layout()
    out_plot = PLOT_DIR / "learning_curve_200.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)

    print(f"\n[완료] 200 Epoch 학습 및 그래프 저장 완료: {out_plot}")
    print(f" - 최적 Epoch: {model.best_epoch} (Val RMSE: {best_rmse:.5f})")
    print(f" - Epoch 200 시점 Train RMSE: {history[-1, 1]:.5f}, Val RMSE: {history[-1, 2]:.5f}")


if __name__ == "__main__":
    main()

