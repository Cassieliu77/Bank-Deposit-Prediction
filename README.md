# 📊 Customer Subscription Prediction for Bank Marketing: A Tradeoff Between Recall and Targeting Efficiency

## Overview

This project predicts whether a bank customer will subscribe to a term deposit using machine learning classification models. The goal is to improve bank marketing efficiency by identifying high-potential customers while balancing customer coverage and outreach efficiency.

The project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training from scratch using NumPy
- Model evaluation and comparison
- Streamlit deployment for real-time prediction

---

## Dataset

This project uses the UCI Bank Marketing Dataset, which contains customer demographic, financial, and campaign-related information collected from a Portuguese bank.

- ~45,000 customer records
- Numerical and categorical features
- Binary target variable: term deposit subscription (`yes` / `no`)

---

## Repository Contents

```bash
.
├── home.py                                   # Main Streamlit application
├── pages/                                    # Additional Streamlit pages
├── utils/                                    # Helper functions and utilities
├── Bank_Marketing_Preprocessing.ipynb        # Data cleaning and preprocessing
├── logistic_regression.ipynb                 # Logistic Regression implementation
├── decision_tree_updated.ipynb               # Decision Tree implementation
├── model_comparison.ipynb                    # Model evaluation and comparison
├── cleaned_data.csv                          # Processed dataset
├── README.md
└── LICENSE
```

---

## Machine Learning Pipeline

### Data Processing
The preprocessing pipeline includes:
- Handling missing categorical values
- Removing `duration` to prevent data leakage
- Feature engineering from `pdays`
- One-hot encoding categorical variables
- Standardizing continuous variables
- Handling skewed distributions and outliers

### Models
Two classification models were implemented entirely from scratch using NumPy:
- Logistic Regression
- Decision Tree

### Evaluation Metrics
Because the dataset is highly imbalanced, the project focuses on:
- Precision
- Recall
- F1-score
- F2-score
- AUC-ROC
- AUC-PR
- ALIFT

The project compares the tradeoff between:
- Higher Recall and broader lead capture
- Higher Precision and more efficient customer targeting

---

## Running the Streamlit Application

### 1. Clone the Repository

```bash
git clone <your-repository-link>
cd <repository-name>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install the required packages manually:

```bash
pip install streamlit pandas numpy matplotlib
```

### 3. Run the Application

```bash
streamlit run home.py
```

The Streamlit application allows users to:
- Input customer information
- Apply the preprocessing pipeline
- Generate subscription probability predictions

---

## Key Insights

- Previous campaign success is highly predictive
- Customer contact history strongly affects subscription likelihood
- Logistic Regression captures more potential subscribers through higher Recall
- Decision Tree provides stronger targeting efficiency and reduces unnecessary outreach

---

## Future Improvements

Potential future improvements include:
- Ensemble models such as Random Forest or XGBoost
- Cost-sensitive optimization
- Probability calibration
- Improved Streamlit UI/UX

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Streamlit

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/dJBAHCPL)
# final_project_submissions
