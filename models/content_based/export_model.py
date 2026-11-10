"""최종 선택된 Content-Based 모델(G+T+D k50) 학습 및 정적 파일(.pkl) 추출."""

import pickle
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.content_based.content_based import (
    ITEM_FILE, TfidfContentBased, evaluate, load_split
)

OUT_PATH = Path(__file__).resolve().parent / "cb_model.pkl"


def main():
    train = load_split("train.csv")
    val = load_split("val.csv")
    test = load_split("test.csv")
    items = pd.read_csv(ITEM_FILE)
    n_users = int(train["u"].max()) + 1

    # 튜닝을 통해 최종 선택된 하이퍼파라미터 (G+T+D k50)
    model = TfidfContentBased(
        n_users=n_users,
        items=items,
        features=("genre", "title", "decade"),
        k=50,
        damp=0.5,
    ).fit(train)

    val_rmse, val_mae = evaluate(model, val)
    test_rmse, test_mae = evaluate(model, test)
    print(f"[Content-Based 최종 모델] Val RMSE={val_rmse:.5f}, Test RMSE={test_rmse:.5f}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"정적 모델 저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()

