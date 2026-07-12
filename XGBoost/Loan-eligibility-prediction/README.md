# Loan Approval Prediction Using XGBoost

## Overview

This project develops a Machine Learning model to predict loan approval decisions based on applicant demographics, financial information, and credit history. The workflow includes data preprocessing, exploratory data analysis (EDA), feature engineering, handling class imbalance using SMOTE, model training, hyperparameter tuning, and performance evaluation.

The final model utilizes XGBoost and achieves approximately **84.22% cross-validation accuracy**, providing valuable insights into the factors influencing loan approval decisions.

---

## Problem Statement

Financial institutions process thousands of loan applications and must evaluate applicants efficiently while minimizing default risk.

The objective of this project is to build a predictive model that can determine whether a loan application is likely to be approved based on applicant information.

---

## Dataset Features

| Feature           | Description           |
| ----------------- | --------------------- |
| Gender            | Applicant Gender      |
| Married           | Marital Status        |
| Dependents        | Number of Dependents  |
| Education         | Education Level       |
| Self_Employed     | Employment Status     |
| ApplicantIncome   | Applicant Income      |
| CoapplicantIncome | Co-applicant Income   |
| LoanAmount        | Requested Loan Amount |
| Loan_Amount_Term  | Loan Repayment Term   |
| Credit_History    | Credit History Status |
| Property_Area     | Property Location     |
| Loan_Status       | Target Variable       |

---

## Project Workflow

### 1. Data Preprocessing

* Missing value treatment
* Data type validation
* Duplicate checking
* Feature encoding

### 2. Exploratory Data Analysis (EDA)

* Univariate Analysis
* Bivariate Analysis
* Target distribution analysis
* Correlation analysis

### 3. Feature Engineering

* Label Encoding
* One-Hot Encoding
* Data transformation

### 4. Handling Class Imbalance

* Applied SMOTE (Synthetic Minority Oversampling Technique)
* Balanced the training dataset before model training

### 5. Model Development

Baseline models evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost Classifier

### 6. Hyperparameter Tuning

RandomizedSearchCV was used to optimize:

* n_estimators
* max_depth
* learning_rate
* subsample
* colsample_bytree
* gamma
* min_child_weight

---

## Best XGBoost Parameters

```python
{
    'subsample': 0.9,
    'n_estimators': 100,
    'min_child_weight': 1,
    'max_depth': 8,
    'learning_rate': 0.05,
    'gamma': 0.1,
    'colsample_bytree': 0.8
}
```

---

## Model Performance

| Metric                 | Score                   |
| ---------------------- | ----------------------- |
| Cross Validation Score | 84.22%                  |
| Algorithm              | XGBoost Classifier      |
| Validation Method      | 5-Fold Cross Validation |
| Imbalance Handling     | SMOTE                   |

---

## Feature Importance

Top features influencing loan approval:

| Feature                 | Importance |
| ----------------------- | ---------- |
| Credit_History          | 39.77%     |
| Married                 | 12.79%     |
| Property_Area_Semiurban | 6.96%      |
| Property_Area_Urban     | 6.24%      |
| Loan_Amount_Term        | 5.97%      |

---

## Key Business Insights

* Credit History is the strongest predictor of loan approval decisions.
* Married applicants show a higher likelihood of approval.
* Property location significantly impacts approval outcomes.
* Loan repayment duration influences lending risk assessment.
* Historical repayment behavior is more influential than income alone.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn
* XGBoost
* Joblib

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
Loan-eligibility-prediction/
│
├── Loan_Prediction.ipynb
├── requirements.txt
├── loan_prediction_xgb.pkl
├── README.md
└── dataset/
```

---

## Future Improvements

* Compare XGBoost with CatBoost and LightGBM
* Deploy the model using Streamlit or Flask
* Perform advanced feature engineering
* Implement automated model monitoring

---

## Conclusion

An end-to-end machine learning pipeline was developed for loan approval prediction using XGBoost. By incorporating SMOTE for class balancing and hyperparameter tuning for optimization, the model effectively identifies approval patterns and key risk factors. The results demonstrate the importance of credit history, marital status, and property location in loan approval decisions, providing actionable insights for financial institutions.
