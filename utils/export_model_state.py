import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.model_utils import CONTINUOUS_COLS, sigmoid


def gradient_descent(X, y, learning_rate=0.5, iterations=3000, lambda_=0.01):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))

    pos = (y == 1).sum()
    neg = (y == 0).sum()
    weight_vec = np.where(y == 1, neg / pos, 1.0)

    l2_mask = np.ones((n_features, 1))
    l2_mask[0] = 0.0

    for _ in range(iterations):
        y_hat = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (y_hat - y) * weight_vec) / n_samples
        gradient += (lambda_ / n_samples) * weights * l2_mask
        weights -= learning_rate * gradient

    return weights


def export_model_state():
    base_dir = BASE_DIR
    data_path = base_dir / "cleaned_data.csv"
    artifact_path = base_dir / "artifacts" / "model_state.pkl"

    df = pd.read_csv(data_path)
    X_raw = df.drop(columns=["y", "y_binary"]).copy()
    y = df["y_binary"].values.reshape(-1, 1)

    # Preserve exact training behavior from notebook
    for col in X_raw.columns:
        if X_raw[col].dtype == "bool":
            X_raw[col] = X_raw[col].astype(float)

    feature_columns = list(X_raw.columns)
    X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw.values.astype(float)])

    # Stratified split (80/20)
    pos_idx = np.where(y.flatten() == 1)[0]
    neg_idx = np.where(y.flatten() == 0)[0]

    np.random.seed(42)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_test = int(len(pos_idx) * 0.2)
    n_neg_test = int(len(neg_idx) * 0.2)

    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])

    X_train = X[train_idx].copy()

    # Standardization on train only
    col_names = feature_columns
    cont_idx = [col_names.index(c) + 1 for c in CONTINUOUS_COLS if c in col_names]
    train_mean = X_train[:, cont_idx].mean(axis=0)
    train_std = X_train[:, cont_idx].std(axis=0) + 1e-8
    X_train[:, cont_idx] = (X_train[:, cont_idx] - train_mean) / train_std

    y_train = y[train_idx]
    weights = gradient_descent(X_train, y_train, learning_rate=0.5, iterations=3000, lambda_=0.01)

    model_state = {
        "weights": weights.flatten().tolist(),
        "feature_columns": feature_columns,
        "continuous_cols": CONTINUOUS_COLS,
        "train_mean": train_mean.tolist(),
        "train_std": train_std.tolist(),
        "preprocess_config": {
            "campaign_clip_min": float(df["campaign"].min()),
            "campaign_clip_max": float(df["campaign"].max()),
            # Based on UCI Bank Marketing dataset minimum balance used in notebook shift transform.
            "balance_shift": 8019.0,
        },
        "threshold": 0.4,
        "random_seed": 42,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as f:
        pickle.dump(model_state, f)

    print(f"Exported model state to: {artifact_path}")


if __name__ == "__main__":
    export_model_state()
