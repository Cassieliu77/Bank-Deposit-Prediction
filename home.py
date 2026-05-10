import streamlit as st


st.set_page_config(page_title="Bank Deposit Prediction", layout="centered")

st.title("Bank Deposit Prediction")
st.markdown(
    """
This is a minimal Streamlit interface for bank marketing project.

### Context placeholder
- A simple landing page
- A logistic regression **single-sample prediction** page
- A decision tree **single-sample prediction** page

### Member placeholder
- Camile Tong, zt275


### How to use
1. Open the left sidebar and go to **Logistic Regression Predict**
2. Open **Decision Tree Predict** for tree-based scoring
3. Fill the form inputs
4. Click **Predict**
"""
)

st.info(
    "Current version uses a pre-exported model artifact from your notebook training logic."
)
