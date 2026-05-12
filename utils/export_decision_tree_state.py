import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.decision_tree_utils import DecisionTree, fbeta_score, tune_threshold_for_f2
from utils.model_utils import DEFAULT_BALANCE_SHIFT


def stratified_train_val_test_split(y, random_seed=42):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    np.random.seed(random_seed)
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    n_pos_test = int(len(pos_idx) * 0.2)
    n_neg_test = int(len(neg_idx) * 0.2)
    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])

    pos_rem = pos_idx[n_pos_test:]
    neg_rem = neg_idx[n_neg_test:]

    n_pos_val = int(len(pos_rem) * 0.2)
    n_neg_val = int(len(neg_rem) * 0.2)
    val_idx = np.concatenate([pos_rem[:n_pos_val], neg_rem[:n_neg_val]])
    train_idx = np.concatenate([pos_rem[n_pos_val:], neg_rem[n_neg_val:]])

    return train_idx, val_idx, test_idx


def export_decision_tree_state():
    data_path = BASE_DIR / "cleaned_data.csv"
    artifact_path = BASE_DIR / "artifacts" / "decision_tree_state.pkl"

    df = pd.read_csv(data_path)
    X_raw = df.drop(columns=["y", "y_binary"]).copy()
    y_all = df["y_binary"].values.astype(int)

    for col in X_raw.columns:
        if X_raw[col].dtype == "bool":
            X_raw[col] = X_raw[col].astype(float)

    feature_columns = list(X_raw.columns)
    X_all = X_raw.values.astype(float)

    train_idx, val_idx, _ = stratified_train_val_test_split(y_all, random_seed=42)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]

    depths = [7, 9, 11]
    min_samples_options = [10, 20, 50, 100, 200]
    best_val_f2 = -1.0
    best_params = (depths[0], min_samples_options[0])

    for depth in depths:
        for min_samples in min_samples_options:
            tree = DecisionTree(max_depth=depth, min_samples_split=min_samples, class_weight="balanced")
            tree.fit(X_train, y_train)
            y_val_pred = tree.predict_with_threshold(X_val, threshold=0.5)
            val_f2 = fbeta_score(y_val, y_val_pred, beta=2.0)
            if val_f2 > best_val_f2:
                best_val_f2 = val_f2
                best_params = (depth, min_samples)

    best_depth, best_min_samples = best_params
    final_tree = DecisionTree(
        max_depth=best_depth,
        min_samples_split=best_min_samples,
        class_weight="balanced",
    )
    final_tree.fit(X_train, y_train)

    val_proba = final_tree.predict_proba(X_val)
    best_threshold, best_threshold_f2 = tune_threshold_for_f2(val_proba, y_val, n_steps=200)

    model_state = {
        "tree": final_tree.tree,
        "feature_columns": feature_columns,
        "threshold": float(best_threshold),
        "threshold_metric": "f2_beta_2_on_validation",
        "threshold_f2": float(best_threshold_f2),
        "hyperparameters": {
            "max_depth": int(best_depth),
            "min_samples_split": int(best_min_samples),
            "class_weight": "balanced",
        },
        "preprocess_config": {
            "campaign_clip_min": float(df["campaign"].min()),
            "campaign_clip_max": float(df["campaign"].max()),
            "balance_shift": float(DEFAULT_BALANCE_SHIFT),
        },
        "random_seed": 42,
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as f:
        pickle.dump(model_state, f)

    print(f"Exported decision tree state to: {artifact_path}")
    print(
        "Selected hyperparameters:"
        f" max_depth={best_depth}, min_samples_split={best_min_samples};"
        f" threshold={best_threshold:.3f} (val F2={best_threshold_f2:.4f})"
    )


if __name__ == "__main__":
    export_decision_tree_state()
