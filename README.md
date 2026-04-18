# 📊 Bank Deposit Prediction

## 📌 Overview  
This project aims to improve bank marketing efficiency by predicting whether a customer will subscribe to a term deposit. Direct marketing campaigns are costly and often involve contacting many uninterested customers. By leveraging machine learning, this project helps identify high-potential customers and optimize resource allocation.

The project builds an end-to-end pipeline, including data exploration, preprocessing, feature engineering, model development, and deployment using a web-based interface.

---

## 🎯 Objectives  
- Predict customer subscription to term deposits (binary classification)  
- Improve marketing efficiency by reducing unnecessary outreach  
- Build interpretable models for business decision-making  
- Deploy a simple application for real-time prediction  

---

## 📂 Dataset  
We use the UCI Bank Marketing Dataset, which contains customer demographic, financial, and campaign-related information collected from a Portuguese bank.

- Number of samples: ~45,000  
- Features: Demographic, financial, and campaign data  
- Target: Whether the client subscribed to a term deposit (`yes` / `no`)  

---

## 🔍 Data Processing Pipeline  

### 1. Exploratory Data Analysis (EDA)
- Examined class imbalance (low subscription rate)  
- Analyzed distributions of numerical features  
- Explored relationships between features and target  

### 2. Data Cleaning
- Handled missing values by treating them as informative categories  
- Removed `duration` to prevent data leakage  
- Capped outliers in `balance` using IQR method  

### 3. Feature Engineering
- Converted binary variables (`yes/no`) to numeric format  
- Created `previous_contact` from `pdays`  
- Constructed `pdays_clean` to handle special values (-1)  

### 4. Data Preprocessing
- One-hot encoding for categorical variables  
- Min-Max normalization for numerical features  

---

## 🤖 Models  

**TO BE EDITED**

---

## 📊 Evaluation Metrics  

Due to class imbalance, we focus on:

- **Precision** – reduce wasted marketing effort  
- **Recall** – identify potential customers  
- **F1-score** – balance between precision and recall  
- Accuracy

---

## 🌐 Deployment  

A web application is built using Streamlit:

- Input customer features  
- Apply preprocessing pipeline  
- Generate prediction probability  
- Provide recommendation (e.g., High / Low potential)  

---

## 🧠 Key Insights  
- Customer contact history is highly predictive  
- Financial indicators (e.g., balance) influence subscription likelihood  
- Proper feature engineering significantly improves model performance  

---

## ⚠️ Challenges & Solutions  

| Challenge | Solution |
|----------|--------|
| Class imbalance | Use F1-score and Recall |
| Data leakage | Remove `duration` |
| Mixed feature types | Apply encoding and normalization |
| Special values (`pdays = -1`) | Feature engineering |

---

## 🚀 Future Improvements  
- Add more advanced models (e.g., XGBoost)  
- Perform hyperparameter tuning  
- Incorporate cost-sensitive learning  
- Improve UI/UX of the application  

---

## 📁 Project Structure  
**TO BE EDITED**

