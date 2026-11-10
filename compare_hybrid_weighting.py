"""어셈블 가중치 방식 비교 분석 및 시각화 스크립트 (compare_hybrid_weighting.py)
1. 가중치 함수 곡선 비교: 일반 선형 vs 곡선(베이지안 스무딩) (X축: 평가 데이터 수 N_u, Y축: CB 모델 반영 비율 w_cb)
2. Test 데이터셋 최종 성능 비교: 선형 가중치 하이브리드 vs 곡선 최적화 하이브리드
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
TAU = 5.0  # 곡선 방식 최적 파라미터


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


def calc_metrics(actual, pred):
    err = actual - pred
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def plot_weight_functions(max_n=150):
    """1. X축: 평가 데이터 수(N_u), Y축: CB 반영 비율(w_cb) 곡선 vs 선형 플롯"""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    n_vals = np.linspace(0, max_n, 300)

    # 1) 제안 곡선 (베이지안 스무딩): w_cb = tau / (N_u + tau)
    w_cb_curve = TAU / (n_vals + TAU)

    # 2) 일반 선형 (Linear Decay): N_u가 0일 때 1.0, 훈련 데이터 중앙값(46) 또는 50에서 0으로 떨어지는 선형
    k_max = 50.0  # 50개 평가 시 CB 비중 0% (선형 감쇄 기준점)
    w_cb_linear = np.clip(1.0 - (n_vals / k_max), 0.0, 1.0)

    # 3) 단순 최대치 선형 (전체 최대 평가수 516 기준)
    w_cb_linear_global = np.clip(1.0 - (n_vals / 516.0), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(n_vals, w_cb_curve, label=f"Proposed Curve (tau={TAU})", color="#d62728", linewidth=2.5)
    ax.plot(n_vals, w_cb_linear, label="Linear Decay (0 ~ 50 ratings)", color="#1f77b4", linestyle="--", linewidth=2.0)
    ax.plot(n_vals, w_cb_linear_global, label="Global Linear (0 ~ 516 ratings)", color="#7f7f7f", linestyle=":", linewidth=1.8)

    ax.set_title("Content-Based Weight (w_CB) by User Rating Count (N_u)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of User Ratings (N_u)", fontsize=11)
    ax.set_ylabel("CB Model Weight (w_CB)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max_n)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10.5)

    fig.tight_layout()
    out_file = PLOT_DIR / "hybrid_weight_curve_comparison.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"[플롯 1 저장 완료] 가중치 함수 비교 플롯: {out_file}")


def evaluate_and_plot_performance(cb_model, mf_model, test, user_count):
    """2. 선형 vs 곡선 어셈블 모델의 Test 성능 비교 막대그래프"""
    u_test = test["u"].to_numpy(np.int64)
    i_test = test["i"].to_numpy(np.int64)
    y_test = test["rating"].to_numpy(np.float64)

    test_cb = cb_model.predict(u_test, i_test)
    test_cf = mf_model.predict(u_test, i_test)
    nu = user_count[u_test]

    # 1) 곡선 방식 (Proposed Curve, tau=5)
    w_cf_curve = nu / (nu + TAU)
    w_cb_curve = 1.0 - w_cf_curve
    pred_curve = np.clip(w_cb_curve * test_cb + w_cf_curve * test_cf, 1.0, 5.0)

    # 2) 선형 감쇄 방식 (Linear, k_max=50)
    w_cf_linear = np.clip(nu / 50.0, 0.0, 1.0)
    w_cb_linear = 1.0 - w_cf_linear
    pred_linear = np.clip(w_cb_linear * test_cb + w_cf_linear * test_cf, 1.0, 5.0)

    # 3) 단순 5:5 고정 평균 (Fixed 0.5)
    pred_fixed = np.clip(0.5 * test_cb + 0.5 * test_cf, 1.0, 5.0)

    results = {
        "Fixed Average (5:5)": calc_metrics(y_test, pred_fixed),
        "Linear Weight": calc_metrics(y_test, pred_linear),
        "Proposed Curve (tau=5)": calc_metrics(y_test, pred_curve),
    }

    print("\n[Test 데이터셋 어셈블 가중치 방식별 성능 비교]")
    for name, (rmse, mae) in results.items():
        print(f" - {name:<22}: RMSE = {rmse:.5f}, MAE = {mae:.5f}")

    # 막대그래프 생성
    names = list(results.keys())
    rmses = [results[k][0] for k in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(names, rmses, color=["#7f7f7f", "#1f77b4", "#d62728"], width=0.45)
    ax.set_title("Assembly Weighting Strategy Comparison on Test Set", fontsize=12, fontweight="bold")
    ax.set_ylabel("Test RMSE (lower is better)", fontsize=11)
    ax.set_ylim(min(rmses) - 0.015, max(rmses) + 0.01)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val:.5f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_file = PLOT_DIR / "hybrid_weight_performance_comparison.png"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"[플롯 2 저장 완료] 성능 비교 막대그래프: {out_file}")


def main():
    cb_model, mf_model = load_models()
    test = pd.read_csv(DATA_DIR / "test.csv")
    user_count = np.load(DATA_DIR / "train_stats.npz")["user_count"]

    # 1. 가중치 곡선 플롯 (X: 평가수 N_u, Y: CB 반영비율)
    plot_weight_functions(max_n=100)

    # 2. Test 성능 비교 플롯
    evaluate_and_plot_performance(cb_model, mf_model, test, user_count)


if __name__ == "__main__":
    main()

