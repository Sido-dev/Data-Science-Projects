# Loan Default Prediction using LightGBM

## Overview

This project aims to predict whether a borrower will default on a loan using demographic, employment, financial, and credit-related information. The objective is to build a robust classification model that can assist financial institutions in identifying high-risk borrowers and making informed lending decisions.

LightGBM was selected as the primary algorithm due to its efficiency, scalability, and strong performance on large structured datasets.

---

## Dataset

The dataset contains borrower information including:

### Numerical Features
- Age
- Income
- LoanAmount
- CreditScore
- MonthsEmployed
- NumCreditLines
- InterestRate
- LoanTerm
- DTIRatio

### Categorical Features
- Education
- EmploymentType
- MaritalStatus
- HasMortgage
- HasDependents
- LoanPurpose
- HasCoSigner

### Target Variable
- Default (0 = No Default, 1 = Default)

Dataset Source:

https://www.kaggle.com/datasets/nikhil1e9/loan-default

---

## Project Workflow

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Categorical Encoding
      ↓
Train-Test Split
      ↓
LightGBM Model Training
      ↓
Hyperparameter Tuning
      ↓
Threshold Optimization
      ↓
Feature Importance Analysis
      ↓
SHAP Explainability
      ↓
Model Saving
```

---

## Exploratory Data Analysis

The dataset was analyzed to understand:

- Target class distribution
- Feature distributions
- Correlation between variables
- Potential predictors of loan default

Visualizations included:

- Histograms
- Count plots
- Correlation heatmaps

---

## Feature Engineering

A new feature was created:

```python
loan_income_ratio = LoanAmount / Income
```

This feature helps capture the relationship between the requested loan amount and borrower income.

---

## Data Preprocessing

Categorical variables were encoded using ordinal mappings:

- Education
- EmploymentType
- MaritalStatus
- HasMortgage
- HasDependents
- LoanPurpose
- HasCoSigner

Train-test split:

```python
test_size = 0.20
random_state = 42
stratify = y
```

---

## LightGBM Model

### Initial Model

```python
LGBMClassifier(random_state=42)
```

### Baseline Performance

| Metric | Score |
|----------|----------|
| Accuracy | 0.8863 |
| Precision | 0.5827 |
| Recall | 0.0725 |
| F1 Score | 0.1290 |
| ROC-AUC | 0.7550 |

---

## Hyperparameter Tuning

RandomizedSearchCV was used to optimize:

```python
max_depth
num_leaves
learning_rate
n_estimators
min_child_samples
```

### Best Tuned Performance

| Metric | Score |
|----------|----------|
| Accuracy | 0.8867 |
| Precision | 0.6244 |
| Recall | 0.0614 |
| F1 Score | 0.1118 |
| ROC-AUC | 0.7600 |

### Best ROC-AUC

```text
0.7600
```

The tuned model achieved improved ranking capability as indicated by the ROC-AUC score.

---

## Threshold Optimization

Since loan default prediction is an imbalanced classification problem, threshold tuning was performed.

Instead of the default:

```python
threshold = 0.50
```

multiple thresholds were evaluated:

```python
0.10
0.15
0.20
0.25
0.30
```

The optimal threshold was selected based on F1 Score.

---

## Feature Importance

Feature importance analysis identified the most influential predictors of loan default risk.

Top contributing features included:

- Age
- InterestRate
- LoanAmount
- Income
- CreditScore
- DTIRatio
- MonthsEmployed
- loan_income_ratio

---

## Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to interpret model predictions.

Generated visualizations:

- SHAP Summary Plot
- SHAP Bar Plot
- SHAP Waterfall Plot

These explain:

- Which features influence predictions most
- Whether a feature increases or decreases default risk
- Individual prediction explanations

---

## Model Persistence

The final trained model was saved using Joblib.

```python
joblib.dump(best_model, "loan_default_lightgbm.pkl")
```

Loading the model:

```python
model = joblib.load("loan_default_lightgbm.pkl")
```

---

## Results

| Model | ROC-AUC |
|---------|---------|
| LightGBM | 0.7600 |
| XGBoost | 0.7600 |
| CatBoost | 0.7609 |

CatBoost achieved the highest ROC-AUC score, while LightGBM delivered comparable predictive performance with efficient training.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- LightGBM
- XGBoost
- SHAP
- Joblib

---

## Key Learnings

- Handling imbalanced classification problems
- Feature engineering for credit risk modeling
- Hyperparameter tuning using RandomizedSearchCV
- Threshold optimization for improved recall and F1 score
- Model interpretability using SHAP
- Comparing gradient boosting algorithms for structured data

---

## Author

Sudhanshu Narayane

GitHub: https://github.com/Sido-dev
LinkedIn: https://www.linkedin.com/in/sid101110