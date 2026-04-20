import numpy as np


CONTINUOUS_COLS = ["age", "balance", "day_of_week", "campaign", "pdays_clean", "previous"]
DEFAULT_BALANCE_SHIFT = 8019.0

CATEGORICAL_GROUPS = {
    "job": [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": ["apr", "aug", "dec", "feb", "jan", "jul", "jun", "mar", "may", "nov", "oct", "sep"],
    "poutcome": ["failure", "not_contacted", "other", "success"],
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def transform_raw_business_inputs(
    raw_inputs: dict,
    campaign_clip_min: float,
    campaign_clip_max: float,
    balance_shift: float = DEFAULT_BALANCE_SHIFT,
) -> dict:
    """
    Convert raw business fields to model-space engineered features.
    """
    pdays_raw = float(raw_inputs["pdays_raw"])
    previous_contact = 0.0 if pdays_raw == -1.0 else 1.0
    pdays_clean = 0.0 if pdays_raw == -1.0 else pdays_raw

    campaign_raw = float(raw_inputs["campaign_raw"])
    campaign = float(np.clip(campaign_raw, campaign_clip_min, campaign_clip_max))

    previous_raw = float(raw_inputs["previous_raw"])
    previous = float(np.log1p(max(previous_raw, 0.0)))

    balance_raw = float(raw_inputs["balance_raw"])
    # Keep notebook-style transformation: log(raw - min_balance + 1)
    balance = float(np.log(max(balance_raw + balance_shift, 1.0)))

    return {
        "age": float(raw_inputs["age"]),
        "default": float(raw_inputs["default"]),
        "housing": float(raw_inputs["housing"]),
        "loan": float(raw_inputs["loan"]),
        "day_of_week": float(raw_inputs["day_of_week"]),
        "campaign": campaign,
        "previous": previous,
        "previous_contact": previous_contact,
        "pdays_clean": pdays_clean,
        "balance": balance,
    }


def standardize_continuous(features: dict, train_mean: np.ndarray, train_std: np.ndarray) -> dict:
    standardized = dict(features)
    for idx, col in enumerate(CONTINUOUS_COLS):
        standardized[col] = (float(features[col]) - float(train_mean[idx])) / float(train_std[idx])
    return standardized


def build_feature_vector(
    standardized_features: dict,
    selected_categories: dict,
    feature_columns: list[str],
) -> np.ndarray:
    row = {col: 0.0 for col in feature_columns}

    # Numeric and binary features
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
        row[base_col] = float(standardized_features[base_col])

    # One-hot categorical features
    for group_name, selected_value in selected_categories.items():
        col_name = f"{group_name}_{selected_value}"
        if col_name in row:
            row[col_name] = 1.0

    vector = np.array([row[col] for col in feature_columns], dtype=float)
    # Add intercept term at index 0
    return np.hstack(([1.0], vector))


def predict_probability(feature_vector: np.ndarray, weights: np.ndarray) -> float:
    z = np.dot(feature_vector, weights).item()
    return float(sigmoid(np.array([z]))[0])
