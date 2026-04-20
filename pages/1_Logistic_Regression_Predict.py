import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from utils.model_utils import (
    CATEGORICAL_GROUPS,
    predict_probability,
    standardize_continuous,
    build_feature_vector,
    transform_raw_business_inputs,
)


st.title("Logistic Regression Predict")
st.caption("Single-sample prediction using pre-exported model weights.")

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_STATE_PATH = BASE_DIR / "artifacts" / "model_state.pkl"


@st.cache_resource
def load_model_state(path: Path):
    with path.open("rb") as f:
        state = pickle.load(f)
    state["weights"] = np.array(state["weights"], dtype=float)
    state["train_mean"] = np.array(state["train_mean"], dtype=float)
    state["train_std"] = np.array(state["train_std"], dtype=float)
    return state


if not MODEL_STATE_PATH.exists():
    st.error(
        "Missing model artifact: artifacts/model_state.pkl. "
        "Run `python utils/export_model_state.py` first."
    )
    st.stop()

state = load_model_state(MODEL_STATE_PATH)
pre_cfg = state.get("preprocess_config", {})
campaign_clip_min = float(pre_cfg.get("campaign_clip_min", 1.0))
campaign_clip_max = float(pre_cfg.get("campaign_clip_max", 6.0))
balance_shift = float(pre_cfg.get("balance_shift", 8019.0))


def half_width_number_input(label: str, **kwargs):
    col, _ = st.columns([1, 1])
    with col:
        return st.number_input(label, **kwargs)


def half_width_selectbox(label: str, options, index=0):
    col, _ = st.columns([1, 1])
    with col:
        return st.selectbox(label, options=options, index=index)

with st.form("single_predict_form"):
    st.subheader("Inputs")
    age = half_width_number_input("Age", min_value=16.0, max_value=100.0, value=40.0, step=1.0)
    housing = half_width_selectbox("Has Housing Loan", options=[0, 1], index=1)
    loan = half_width_selectbox("Has Personal Loan", options=[0, 1], index=0)
    balance_raw = half_width_number_input("Balance", value=1500.0, step=100.0)
    campaign_raw = half_width_number_input("CurrentCampaign Contacts", min_value=1.0, value=2.0, step=1.0)
    previous_raw = half_width_number_input("Previous Contacts", min_value=0.0, value=0.0, step=1.0)
    pdays_raw = half_width_number_input("Days Since Last Contact", value=-1.0, step=1.0)

    contact = half_width_selectbox("Contact Type", options=CATEGORICAL_GROUPS["contact"], index=0)
    month = half_width_selectbox("Last Contact Month", options=CATEGORICAL_GROUPS["month"], index=8)
    poutcome = half_width_selectbox("Previous Outcome", options=CATEGORICAL_GROUPS["poutcome"], index=1)

    with st.expander("Advanced Inputs", expanded=False):
        default = half_width_selectbox("Has Credit Default", options=[0, 1], index=0)
        day_of_week = half_width_number_input("Day of Month Contacted", min_value=1.0, max_value=31.0, value=15.0, step=1.0)
        job = half_width_selectbox("Job", options=CATEGORICAL_GROUPS["job"], index=0)
        marital = half_width_selectbox("Marital Status", options=CATEGORICAL_GROUPS["marital"], index=1)
        education = half_width_selectbox("Education", options=CATEGORICAL_GROUPS["education"], index=1)
        threshold = st.slider(
            "Decision threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(state["threshold"]),
            step=0.05,
        )

    selected_categories = {
        "job": job,
        "marital": marital,
        "education": education,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
    }

    st.caption(
        f"Raw-to-model transforms: campaign clipped to [{campaign_clip_min:.0f}, {campaign_clip_max:.0f}], "
        f"previous uses log1p, pdays -1 => no previous contact, balance uses shifted log (shift={balance_shift:.0f})."
    )
    submitted = st.form_submit_button("Predict")

if submitted:
    raw_features = {
        "age": age,
        "default": default,
        "housing": housing,
        "loan": loan,
        "day_of_week": day_of_week,
        "campaign_raw": campaign_raw,
        "previous_raw": previous_raw,
        "pdays_raw": pdays_raw,
        "balance_raw": balance_raw,
    }

    model_space_features = transform_raw_business_inputs(
        raw_inputs=raw_features,
        campaign_clip_min=campaign_clip_min,
        campaign_clip_max=campaign_clip_max,
        balance_shift=balance_shift,
    )
    standardized = standardize_continuous(model_space_features, state["train_mean"], state["train_std"])
    feature_vector = build_feature_vector(standardized, selected_categories, state["feature_columns"])
    probability = predict_probability(feature_vector, state["weights"])
    pred_label = 1 if probability >= threshold else 0

    st.success(f"Predicted probability (subscribe=yes): **{probability:.4f}**")
    st.write(f"Predicted class at threshold {threshold:.2f}: **{pred_label}**")
    if pred_label == 1:
        st.info("Interpretation: likely to subscribe.")
    else:
        st.info("Interpretation: likely not to subscribe.")
