"""하이브리드 가중치 비율(tau) 튜닝 및 최종 성능 평가 스크립트 (tune_hybrid.py)
- Validation 데이터셋에서 최적의 가중치 기준값(tau) 탐색
- Test 데이터셋에서 단일 모델(CB, SVD) vs 하이브리드 모델 성능 비교 및 그래프 저장
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.content_based.export_model import OUT_PATH as CB_PKL, main as export_cb
from models.model_based.export_model import OUT_PATH as MF_PKL, main as export_mf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "100k" / "preprocessed"
PLOT_DIR = ROOT / "plots"
TAU_CANDIDATES = [1.0, 2.0, 5.0, 10.0, 20.0, 46.0]


def load_models():
    if not CB_PKL.exists():
        export_cb()
    if not MF_PKL.exists():
        export_mf()
    with open(CB_PKL, "rb") as f:
        cb_model = pickle.load(f)
    with open(MF_PKL, "rb") as f:
        mf_model = pickle.load(f)
    return cb_model, mf_model


def get_weights(n_u, tau):
    """유저 평점 수(N_u)에 따른 CB / CF 가중치 비율 계산."""
    w_cf = n_u / (n_u + tau)
    w_cb = 1.0 - w_cf
    return w_cb, w_cf


def calc_metrics(actual, pred):
    err = actual - pred
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def save_plot(results):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    rmses = [results[k][0] for k in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, rmses, color=["#55A868", "#4C72B0", "#C44E52"], width=0.5)
    ax.set(ylabel="Test RMSE (lower is better)", title="Model Comparison on Test Set")
    ax.set_ylim(min(rmses) - 0.03, max(rmses) + 0.02)
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.5f}", ha="center")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hybrid_comparison.png", dpi=150)
    plt.close(fig)


def main():
    cb_model, mf_model = load_models()
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    user_count = np.load(DATA_DIR / "train_stats.npz")["user_count"]

    # 1. Validation 세트로 최적의 가중치 비율 기준값(tau) 튜닝
    u_val, i_val = val["u"].to_numpy(np.int64), val["i"].to_numpy(np.int64)
    y_val = val["rating"].to_numpy(np.float64)
    val_cb, val_cf = cb_model.predict(u_val, i_val), mf_model.predict(u_val, i_val)

    print("[1. Validation 데이터셋 하이브리드 비율(tau) 튜닝]")
    best_tau, best_val_rmse = None, float("inf")
    for tau in TAU_CANDIDATES:
        w_cb, w_cf = get_weights(user_count[u_val], tau)
        pred = np.clip(w_cb * val_cb + w_cf * val_cf, 1.0, 5.0)
        rmse, mae = calc_metrics(y_val, pred)
        print(f" - tau={tau:<4} -> Val RMSE = {rmse:.5f}, Val MAE = {mae:.5f}")
        if rmse < best_val_rmse:
            best_val_rmse, best_tau = rmse, tau

    print(f"=> 최적 비율 파라미터 선정: tau = {best_tau} (Val RMSE = {best_val_rmse:.5f})")

    # 2. 선정된 최적 tau로 Test 데이터셋 최종 성능 비교
    u_test, i_test = test["u"].to_numpy(np.int64), test["i"].to_numpy(np.int64)
    y_test = test["rating"].to_numpy(np.float64)
    test_cb, test_cf = cb_model.predict(u_test, i_test), mf_model.predict(u_test, i_test)
    w_cb, w_cf = get_weights(user_count[u_test], best_tau)
    test_hyb = np.clip(w_cb * test_cb + w_cf * test_cf, 1.0, 5.0)

    results = {
        "Content-Based": calc_metrics(y_test, test_cb),
        "Model-Based (SVD)": calc_metrics(y_test, test_cf),
        "Hybrid": calc_metrics(y_test, test_hyb),
    }

    print(f"\n[2. Test 데이터셋 최종 성능 비교 (tau={best_tau})]")
    for name, (rmse, mae) in results.items():
        print(f" - {name:<18}: Test RMSE = {rmse:.5f}, Test MAE = {mae:.5f}")

    save_plot(results)
    print(f"\n비교 그래프 저장 완료: {PLOT_DIR / 'hybrid_comparison.png'}")


if __name__ == "__main__":
    main()

