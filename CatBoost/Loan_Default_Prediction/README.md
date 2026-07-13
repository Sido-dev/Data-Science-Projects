# Loan Default Prediction Using CatBoost

## Overview

This project aims to predict whether a borrower is likely to default on a loan using demographic, employment, financial, and credit-related information. The objective is to help financial institutions assess credit risk and make informed lending decisions.

The project follows a complete Machine Learning workflow, including data preprocessing, exploratory data analysis, model training, hyperparameter tuning, threshold optimization, model evaluation, and explainability using SHAP.


## Dataset

The dataset used in this project is publicly available on Kaggle.

**Dataset:** [Loan Default Prediction Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

### Download

Download the dataset directly from the Kaggle link above and place the `Loan_default.csv` file in the project directory before running the notebook. The dataset contains **255,347 records** and **18 features**, including demographic, financial, employment, and loan-related attributes used to predict loan default.


## Problem Statement

Loan defaults can result in significant financial losses for lending institutions. Accurately identifying high-risk borrowers enables organizations to reduce risk exposure and improve decision-making.

The goal of this project is to build a classification model capable of predicting loan default status based on borrower characteristics.

---

## Dataset Information

The dataset contains both numerical and categorical features related to borrower demographics, employment status, financial history, and loan details.

### Target Variable

* **Default**

  * 0 → No Default
  * 1 → Default

### Numerical Features

* Age
* Income
* LoanAmount
* CreditScore
* MonthsEmployed
* NumCreditLines
* InterestRate
* LoanTerm
* DTIRatio

### Categorical Features

* Education
* EmploymentType
* MaritalStatus
* HasMortgage
* HasDependents
* LoanPurpose
* HasCoSigner

---

## Project Workflow

### 1. Data Preprocessing

* Data loading and inspection
* Missing value analysis
* Data quality checks
* Feature selection

### 2. Exploratory Data Analysis (EDA)

* Distribution analysis
* Correlation analysis
* Class imbalance assessment
* Feature relationship analysis

### 3. Train-Test Split

* Stratified train-test split
* Preserved class distribution across datasets

### 4. Model Development

A CatBoost Classifier was selected due to its ability to handle categorical variables natively without extensive encoding.

### 5. Hyperparameter Tuning

RandomizedSearchCV was used to optimize model parameters including:

* Iterations
* Depth
* Learning Rate
* L2 Regularization

### 6. Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix

### 7. Threshold Optimization

The default classification threshold (0.50) was adjusted to improve minority class detection.

Best threshold identified:

```text
0.20
```

### 8. Explainable AI (SHAP)

SHAP was used to interpret model predictions and identify the most influential features affecting loan default risk.

---

## Model Performance

### Final CatBoost Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 0.8868 |
| Precision | 0.6143 |
| Recall    | 0.0679 |
| F1 Score  | 0.1224 |
| ROC-AUC   | 0.7609 |

### Threshold Optimization

| Threshold | F1 Score |
| --------- | -------- |
| 0.50      | 0.1224   |
| 0.20      | 0.3750   |

Threshold tuning significantly improved the model's ability to identify loan defaulters.

---

## Feature Importance

The most influential features identified by the CatBoost model are:

* Age
* InterestRate
* loan_income_ratio
* MonthsEmployed
* CreditScore

### Top 10 Important Features

| Feature           | Importance |
| ----------------- | ---------: |
| Age               |      30.78 |
| InterestRate      |      21.28 |
| loan_income_ratio |      18.18 |
| MonthsEmployed    |      11.84 |
| CreditScore       |       2.55 |
| EmploymentType    |       2.55 |
| HasCoSigner       |       1.89 |
| HasDependents     |       1.78 |
| NumCreditLines    |       1.45 |
| DTIRatio          |       1.40 |

These results indicate that borrower age, interest rate, loan-to-income relationship, and employment history were the most significant factors influencing loan default predictions. The engineered feature **loan_income_ratio** emerged as one of the strongest predictors, highlighting the importance of feature engineering in improving model performance and credit risk assessment.


---

## Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* CatBoost
* SHAP
* Joblib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Sido-dev/Loan-Default-Prediction.git
```

Move into the project directory:

```bash
cd Loan-Default-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and execute all cells.

---

## Project Structure

```text
Loan-Default-Prediction/
│
├── best_threshold.pkl
├── cat_features.pkl
├── loan_default_catboost.pkl
├── requirements.txt
├── README.md
└── Loan_Default_Prediction.ipynb
```

### File Description

* **Loan_Default_Prediction.ipynb** – Complete notebook containing data preprocessing, EDA, model training, hyperparameter tuning, threshold optimization, feature importance analysis, and SHAP explainability.
* **loan_default_catboost.pkl** – Trained CatBoost model saved using Joblib.
* **cat_features.pkl** – List of categorical features used during model training.
* **best_threshold.pkl** – Optimized classification threshold (0.20) used for final predictions.
* **requirements.txt** – Project dependencies and required Python libraries.
* **README.md** – Project documentation and usage instructions.

---

## Conclusion

A CatBoost-based loan default prediction model was developed to identify borrowers at risk of default. Through hyperparameter tuning and threshold optimization, the model achieved a ROC-AUC score of 0.761 and significantly improved minority-class detection. Feature importance and SHAP analysis provided valuable insights into the factors influencing default risk, making the model both effective and interpretable for credit risk assessment.

---

## Future Improvements

* Compare CatBoost with XGBoost and LightGBM
* Advanced feature engineering
* Model deployment using Flask or FastAPI
* Real-time loan risk prediction dashboard
* Automated model monitoring and retraining

---

## Author

**Sudhanshu Narayane**

Data Science | Machine Learning | AI Enthusiast
