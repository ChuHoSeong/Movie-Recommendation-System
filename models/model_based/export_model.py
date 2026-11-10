"""최종 선택된 Model-Based CF 모델(MF-50-reg) 학습 및 정적 파일(.pkl) 추출."""

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model_based.model_based import BiasedMF, evaluate, load_split

OUT_PATH = Path(__file__).resolve().parent / "mf_model.pkl"


def main():
    train = load_split("train.csv")
    val = load_split("val.csv")
    test = load_split("test.csv")
    n_users = int(train["u"].max()) + 1
    n_items = int(train["i"].max()) + 1

    # 튜닝을 통해 최종 선택된 하이퍼파라미터 (MF-50-reg)
    model = BiasedMF(
        n_users=n_users,
        n_items=n_items,
        n_factors=50,
        n_epochs=50,
        lr=0.005,
        reg=0.05,
    ).fit(train, val)

    val_rmse, val_mae = evaluate(model, val)
    test_rmse, test_mae = evaluate(model, test)
    print(f"[Model-Based 최종 모델] best_epoch={model.best_epoch}, "
          f"Val RMSE={val_rmse:.5f}, Test RMSE={test_rmse:.5f}")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"정적 모델 저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()

