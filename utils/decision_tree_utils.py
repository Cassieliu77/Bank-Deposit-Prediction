import numpy as np

from utils.model_utils import DEFAULT_BALANCE_SHIFT, transform_raw_business_inputs


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10, class_weight=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.tree = None
        self.class_weight_ = {}

    def gini(self, y, w=None):
        if len(y) == 0:
            return 0.0
        if w is None:
            w = np.ones(len(y))
        total_w = w.sum()
        if total_w == 0:
            return 0.0
        p1 = w[y == 1].sum() / total_w
        p0 = 1.0 - p1
        return 1.0 - (p0**2 + p1**2)

    def split_dataset(self, X, y, feature, threshold, w=None):
        mask_left = X[:, feature] <= threshold
        mask_right = ~mask_left
        if w is not None:
            return (
                X[mask_left],
                y[mask_left],
                w[mask_left],
                X[mask_right],
                y[mask_right],
                w[mask_right],
            )
        return (
            X[mask_left],
            y[mask_left],
            None,
            X[mask_right],
            y[mask_right],
            None,
        )

    def best_split(self, X, y, w):
        _, n_features = X.shape
        w_total = w.sum()
        best_gini = float("inf")
        best_info = None

        for feature in range(n_features):
            values = X[:, feature]
            order = np.argsort(values, kind="mergesort")
            x_sorted = values[order]
            y_sorted = y[order]
            w_sorted = w[order]

            if x_sorted[0] == x_sorted[-1]:
                continue

            pos_w = np.where(y_sorted == 1, w_sorted, 0.0)
            neg_w = np.where(y_sorted == 0, w_sorted, 0.0)
            left_pos = np.cumsum(pos_w)[:-1]
            left_neg = np.cumsum(neg_w)[:-1]
            left_w = left_pos + left_neg

            total_pos = float(pos_w.sum())
            total_neg = float(neg_w.sum())
            right_pos = total_pos - left_pos
            right_neg = total_neg - left_neg
            right_w = right_pos + right_neg

            valid = x_sorted[:-1] != x_sorted[1:]
            if not np.any(valid):
                continue

            left_p1 = np.divide(left_pos, left_w, out=np.zeros_like(left_pos), where=left_w > 0)
            left_p0 = np.divide(left_neg, left_w, out=np.zeros_like(left_neg), where=left_w > 0)
            right_p1 = np.divide(right_pos, right_w, out=np.zeros_like(right_pos), where=right_w > 0)
            right_p0 = np.divide(right_neg, right_w, out=np.zeros_like(right_neg), where=right_w > 0)

            left_gini = 1.0 - (left_p0**2 + left_p1**2)
            right_gini = 1.0 - (right_p0**2 + right_p1**2)

            gini_split = (left_w / w_total) * left_gini + (right_w / w_total) * right_gini
            gini_split = np.where(valid, gini_split, np.inf)
            best_idx = int(np.argmin(gini_split))
            best_feature_gini = float(gini_split[best_idx])

            if not np.isfinite(best_feature_gini) or best_feature_gini >= best_gini:
                continue

            threshold = float((x_sorted[best_idx] + x_sorted[best_idx + 1]) / 2.0)
            X_left, y_left, w_left, X_right, y_right, w_right = self.split_dataset(
                X, y, feature, threshold, w
            )
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            best_gini = best_feature_gini
            best_info = {
                "feature": feature,
                "threshold": threshold,
                "gini_split": best_feature_gini,
                "X_left": X_left,
                "y_left": y_left,
                "w_left": w_left,
                "X_right": X_right,
                "y_right": y_right,
                "w_right": w_right,
            }
        return best_info

    def build_tree(self, X, y, depth=0, sample_weights=None):
        n_samples = len(y)
        w = sample_weights if sample_weights is not None else np.ones(n_samples)
        majority_label = int(np.bincount(y.astype(int), weights=w).argmax())

        w0 = self.class_weight_.get(0, 1.0)
        w1 = self.class_weight_.get(1, 1.0)
        w_pos = w1 * float((y == 1).sum())
        w_neg = w0 * float((y == 0).sum())
        prob_pos = w_pos / (w_pos + w_neg) if (w_pos + w_neg) > 0 else 0.0

        def make_leaf():
            return {"label": majority_label, "prob": float(prob_pos)}

        if depth >= self.max_depth:
            return make_leaf()
        if n_samples < self.min_samples_split:
            return make_leaf()

        current_gini = self.gini(y, w)
        if current_gini == 0.0:
            return make_leaf()

        split = self.best_split(X, y, w)
        if split is None:
            return make_leaf()

        left_subtree = self.build_tree(split["X_left"], split["y_left"], depth + 1, split["w_left"])
        right_subtree = self.build_tree(
            split["X_right"], split["y_right"], depth + 1, split["w_right"]
        )
        return {
            "feature": split["feature"],
            "threshold": split["threshold"],
            "n_samples": int(n_samples),
            "gini_gain": float(current_gini - split["gini_split"]),
            "left": left_subtree,
            "right": right_subtree,
        }

    def fit(self, X, y):
        n_samples = len(y)
        classes = np.unique(y)

        if self.class_weight == "balanced":
            self.class_weight_ = {int(c): n_samples / (len(classes) * np.sum(y == c)) for c in classes}
        elif isinstance(self.class_weight, dict):
            self.class_weight_ = {int(k): float(v) for k, v in self.class_weight.items()}
        else:
            self.class_weight_ = {int(c): 1.0 for c in classes}

        sample_weights = np.array([self.class_weight_[int(label)] for label in y], dtype=float)
        self.tree = self.build_tree(X, y, depth=0, sample_weights=sample_weights)
        return self

    def predict_proba_one(self, sample):
        return predict_proba_from_tree(self.tree, sample)

    def predict_proba(self, X):
        return predict_proba_batch(self.tree, X)

    def predict_with_threshold(self, X, threshold=0.5):
        return predict_with_threshold(self.predict_proba(X), threshold)


def fbeta_score(y_true, y_pred, beta=2.0):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta_sq = beta**2
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def tune_threshold_for_f2(probabilities, y_true, n_steps=200):
    thresholds = np.linspace(0.0, 1.0, n_steps)
    best_threshold = 0.5
    best_f2 = -1.0
    for threshold in thresholds:
        y_pred = (probabilities >= threshold).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2.0)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = float(threshold)
    return best_threshold, float(best_f2)


def build_decision_tree_feature_vector(model_space_features, selected_categories, feature_columns):
    row = {col: 0.0 for col in feature_columns}

    for base_col in [
        "age",
        "default",
        "housing",
        "loan",
        "day_of_week",
        "campaign",
        "previous",
        "previous_contact",
        "pdays_clean",
        "balance",
    ]:
        if base_col in row:
            row[base_col] = float(model_space_features[base_col])

    for group_name, selected_value in selected_categories.items():
        col_name = f"{group_name}_{selected_value}"
        if col_name in row:
            row[col_name] = 1.0

    return np.array([row[col] for col in feature_columns], dtype=float)


def build_decision_tree_row_from_raw(
    raw_inputs,
    selected_categories,
    feature_columns,
    campaign_clip_min,
    campaign_clip_max,
    balance_shift=DEFAULT_BALANCE_SHIFT,
):
    model_space_features = transform_raw_business_inputs(
        raw_inputs=raw_inputs,
        campaign_clip_min=campaign_clip_min,
        campaign_clip_max=campaign_clip_max,
        balance_shift=balance_shift,
    )
    feature_row = build_decision_tree_feature_vector(
        model_space_features=model_space_features,
        selected_categories=selected_categories,
        feature_columns=feature_columns,
    )
    return feature_row, model_space_features


def predict_proba_from_tree(tree_node, sample):
    node = tree_node
    while "label" not in node:
        if sample[int(node["feature"])] <= float(node["threshold"]):
            node = node["left"]
        else:
            node = node["right"]
    return float(node.get("prob", float(node["label"])))


def predict_proba_batch(tree_node, X):
    return np.array([predict_proba_from_tree(tree_node, sample) for sample in X], dtype=float)


def predict_with_threshold(probabilities, threshold=0.5):
    return (np.array(probabilities, dtype=float) >= float(threshold)).astype(int)
