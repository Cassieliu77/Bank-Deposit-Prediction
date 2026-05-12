import streamlit as st


st.set_page_config(page_title="Bank Deposit Prediction", layout="centered")

st.title("Bank Deposit Prediction")
st.markdown(
    """
This is a Streamlit interface for bank marketing project.

### Context
This project aims to predict whether a bank customer will subscribe to a term deposit using machine learning classification models. Using the UCI Bank Marketing dataset, we developed an end-to-end machine learning pipeline including data exploration, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

### Team
- Camile Tong
- Jade Yang
- Xinyi Yang
- Yongqi Liu
- Yufan Wang
- Zijing Ding



### How to use
1. Open the left sidebar and go to **Logistic Regression Predict**
2. Open **Decision Tree Predict** for tree-based scoring
3. Fill the form inputs
4. Click **Predict**
"""
)

