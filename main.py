"""최종 하이브리드 영화 추천 시스템 실행 스크립트 (main.py)
- 추출된 정적 모델(.pkl)을 불러와 재학습 없이 즉시 추론
- 유저의 평점 수(N_u, 희소성)에 비례하여 Content-Based와 Model-Based(SVD) 가중치 결합
- 입력된 유저에게 미관측 영화 중 Top-K 추천 영화 출력
"""

import argparse
import pickle
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from models.content_based.content_based import GENRES
from models.content_based.export_model import OUT_PATH as CB_PKL, main as export_cb
from models.model_based.export_model import OUT_PATH as MF_PKL, main as export_mf

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "100k" / "preprocessed"
TAU = 5.0  # tune_hybrid.py에서 최종 선정된 최적 가중치 비율 상수


def get_display_width(text):
    """한글/전각 문자는 너비 2, 영문/숫자는 너비 1로 계산."""
    width = 0
    for char in str(text):
        if unicodedata.east_asian_width(char) in ("F", "W"):
            width += 2
        else:
            width += 1
    return width


def pad_str(text, target_width, align="left"):
    """한글과 영문이 섞여도 너비가 딱 맞도록 공백을 채워주는 함수."""
    text = str(text)
    cur_width = get_display_width(text)
    if cur_width > target_width:
        # target_width를 넘어가면 잘라내고 말줄임표 처리
        res = ""
        w = 0
        for char in text:
            char_w = 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
            if w + char_w > target_width - 2:
                break
            res += char
            w += char_w
        res += ".."
        pad = target_width - get_display_width(res)
        return res + " " * max(0, pad)
    
    pad = target_width - cur_width
    if align == "right":
        return " " * pad + text
    elif align == "center":
        left_pad = pad // 2
        right_pad = pad - left_pad
        return " " * left_pad + text + " " * right_pad
    else:
        return text + " " * pad


def load_models():
    """저장된 정적 모델(.pkl) 로드 (파일이 없으면 최초 1회 추출 실행)."""
    if not CB_PKL.exists():
        export_cb()
    if not MF_PKL.exists():
        export_mf()
    with open(CB_PKL, "rb") as f:
        cb_model = pickle.load(f)
    with open(MF_PKL, "rb") as f:
        mf_model = pickle.load(f)
    return cb_model, mf_model


def get_weights(n_u):
    """유저 평점 수(N_u) 기반 동적 가중치: 희소할수록(N_u가 작을수록) w_cb 증가."""
    w_cf = n_u / (n_u + TAU)
    w_cb = 1.0 - w_cf
    return w_cb, w_cf


def show_test_predictions(u, cb_model, mf_model, test, items, user_count, max_show=5):
    """모델에게 숨겨뒀던(test.csv) 영화들에 대해 예측 평점과 실제 평점을 비교 출력."""
    u_test = test[test["u"] == u]
    if u_test.empty:
        return

    test_items = u_test["i"].to_numpy(dtype=np.int64)
    actual_ratings = u_test["rating"].to_numpy(dtype=np.float64)

    r_cb = cb_model.predict(u, test_items)
    r_cf = mf_model.predict(u, test_items)
    n_u = int(user_count[u])
    w_cb, w_cf = get_weights(n_u)
    hybrid_score = np.clip(w_cb * r_cb + w_cf * r_cf, 1.0, 5.0)

    print(f"\n[검증 테스트] 모델에게 숨겨뒀던 시험지(test.csv) 채점 결과 (총 {len(u_test)}편 중 최대 {max_show}편)")
    h_title = pad_str("영화 제목", 38)
    h_genre = pad_str("장르", 22)
    h_act = pad_str("실제평점", 10, "right")
    h_cb = pad_str("CB예측", 8, "right")
    h_cf = pad_str("CF예측", 8, "right")
    h_hyb = pad_str("Hybrid예측", 12, "right")
    h_err = pad_str("오차", 8, "right")
    print(f"{h_title} {h_genre} {h_act} {h_cb} {h_cf} {h_hyb} {h_err}")
    print("-" * 112)

    for idx in range(min(len(u_test), max_show)):
        movie = items.iloc[test_items[idx]]
        genre_str = "/".join([g for g in GENRES if movie[g] == 1][:3])
        title_str = str(movie["title"])
        err = abs(actual_ratings[idx] - hybrid_score[idx])

        c_title = pad_str(title_str, 38)
        c_genre = pad_str(genre_str, 22)
        c_act = pad_str(f"{actual_ratings[idx]:.1f}", 10, "right")
        c_cb = pad_str(f"{r_cb[idx]:.2f}", 8, "right")
        c_cf = pad_str(f"{r_cf[idx]:.2f}", 8, "right")
        c_hyb = pad_str(f"{hybrid_score[idx]:.2f}", 12, "right")
        c_err = pad_str(f"{err:.2f}", 8, "right")

        print(f"{c_title} {c_genre} {c_act} {c_cb} {c_cf} {c_hyb} {c_err}")


def recommend_top_k(u, top_k, cb_model, mf_model, train, test, items, user_count):
    """입력 유저(u)가 아직 보지 않은 영화 중 하이브리드 점수 상위 K개 추천."""
    seen = set(train.loc[train["u"] == u, "i"])
    unseen = np.array([i for i in range(len(items)) if i not in seen], dtype=np.int64)

    r_cb = cb_model.predict(u, unseen)
    r_cf = mf_model.predict(u, unseen)
    n_u = int(user_count[u])
    w_cb, w_cf = get_weights(n_u)
    hybrid_score = w_cb * r_cb + w_cf * r_cf

    top_idx = np.argsort(-hybrid_score)[:top_k]
    rec_items = unseen[top_idx]

    sparsity = (1.0 - n_u / len(items)) * 100
    print(f"\n================================================================================================================")
    print(f" [유저 u={u}] 학습 평점 수 N={n_u}편 (희소도: {sparsity:.1f}%)")
    print(f" -> 적용 가중치: w_CB = {w_cb:.2f} ({w_cb*100:.0f}%), w_CF = {w_cf:.2f} ({w_cf*100:.0f}%)")
    print(f"================================================================================================================")
    
    # 1. 모델 몰래 숨겨뒀던 test.csv 영화들에 대한 채점 결과 출력
    show_test_predictions(u, cb_model, mf_model, test, items, user_count, max_show=5)

    # 2. 미관측 영화 중 Top-K 추천 목록 출력
    print(f"\n[맞춤 추천] 아직 평가하지 않은 미관측 영화 중 Top-{top_k} 추천 목록")
    r_rank = pad_str("순위", 6)
    r_title = pad_str("영화 제목", 38)
    r_genre = pad_str("장르", 22)
    r_cb_h = pad_str("CB", 6, "right")
    r_cf_h = pad_str("CF", 6, "right")
    r_hyb_h = pad_str("Hybrid", 8, "right")
    print(f"{r_rank} {r_title} {r_genre} {r_cb_h} {r_cf_h} {r_hyb_h}")
    print("-" * 92)

    for rank, idx in enumerate(top_idx, 1):
        movie = items.iloc[rec_items[rank - 1]]
        genre_str = "/".join([g for g in GENRES if movie[g] == 1][:3])
        title_str = str(movie["title"])

        c_rank = pad_str(str(rank), 6)
        c_title = pad_str(title_str, 38)
        c_genre = pad_str(genre_str, 22)
        c_cb = pad_str(f"{r_cb[idx]:.2f}", 6, "right")
        c_cf = pad_str(f"{r_cf[idx]:.2f}", 6, "right")
        c_hyb = pad_str(f"{hybrid_score[idx]:.2f}", 8, "right")

        print(f"{c_rank} {c_title} {c_genre} {c_cb} {c_cf} {c_hyb}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=int, default=None, help="추천할 유저 인덱스 (0 ~ 942)")
    parser.add_argument("--top_k", type=int, default=10, help="추천할 영화 개수 K")
    args = parser.parse_args()

    cb_model, mf_model = load_models()
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    items = pd.read_csv(DATA_DIR / "items.csv").sort_values("i").reset_index(drop=True)
    user_count = np.load(DATA_DIR / "train_stats.npz")["user_count"]

    if args.u is not None:
        recommend_top_k(args.u, args.top_k, cb_model, mf_model, train, test, items, user_count)
    else:
        # 기본 데모: 평점이 가장 적은 Light User vs 평점이 많은 Heavy User 비교
        light_u = int(np.argmin(user_count))
        heavy_u = int(np.argmax(user_count))
        recommend_top_k(light_u, args.top_k, cb_model, mf_model, train, test, items, user_count)
        recommend_top_k(heavy_u, args.top_k, cb_model, mf_model, train, test, items, user_count)


if __name__ == "__main__":
    main()
