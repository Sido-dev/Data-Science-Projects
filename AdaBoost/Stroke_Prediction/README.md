# 🧠 Stroke Prediction using AdaBoost Classifier

## 📌 Project Overview

This project aims to predict the likelihood of a patient experiencing a stroke using machine learning techniques. The workflow includes data preprocessing, exploratory data analysis (EDA), handling class imbalance with SMOTE, model training using AdaBoost, hyperparameter tuning, feature importance analysis, and model evaluation.

The objective is to identify high-risk individuals based on demographic and health-related attributes, enabling proactive healthcare interventions.

---

## 🎯 Problem Statement

Stroke is one of the leading causes of death and long-term disability worldwide. Early detection of stroke risk can help healthcare providers take preventive measures and improve patient outcomes.

This project builds a classification model to predict whether a patient is likely to have a stroke based on various medical and lifestyle factors.

---

## 📂 Dataset

**Dataset:** Stroke Prediction Dataset

### Features

* Gender
* Age
* Hypertension
* Heart Disease
* Ever Married
* Work Type
* Residence Type
* Average Glucose Level
* BMI
* Smoking Status

### Target Variable

| Value | Meaning   |
| ----- | --------- |
| 0     | No Stroke |
| 1     | Stroke    |

---

## 🛠️ Project Workflow

### 1. Data Collection

* Load dataset
* Inspect data structure
* Understand feature distributions

### 2. Exploratory Data Analysis (EDA)

* Correlation Heatmap
* Stroke Distribution Analysis
* Age vs Glucose Level Visualization

### 3. Data Preprocessing

* Missing value treatment using median imputation
* Duplicate value check
* Outlier inspection
* One-Hot Encoding for categorical variables

### 4. Feature Selection

* Separate features and target variable
* Train-Test Split

### 5. Handling Class Imbalance

The dataset is highly imbalanced.

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the minority class before model training.

### 6. Model Building

* AdaBoost Classifier
* Baseline Model
* Hyperparameter Tuning using RandomizedSearchCV

### 7. Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score

### 8. Feature Importance Analysis

Identify the most influential features contributing to stroke prediction.

### 9. Model Saving

The final trained model can be serialized using Joblib.

---

## 🤖 Model Used

### AdaBoost Classifier

AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak learners to create a stronger predictive model.

Benefits:

* Improves predictive performance
* Reduces bias
* Works well with structured/tabular datasets

---

## 🔍 Hyperparameter Tuning

RandomizedSearchCV was used to optimize:

```python
{
    "n_estimators": [50, 100, 200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1]
}
```

### Best Parameters

```python
{
    "n_estimators": 300,
    "learning_rate": 1
}
```

---

## 📊 Model Performance

### Baseline AdaBoost

| Metric          | Score |
| --------------- | ----- |
| Accuracy        | 0.816 |
| Stroke Recall   | 0.42  |
| Stroke F1-Score | 0.18  |

### Tuned AdaBoost

| Metric          | Score |
| --------------- | ----- |
| Accuracy        | 0.952 |
| Stroke Recall   | 0.02  |
| Stroke F1-Score | 0.04  |

> Note: Due to severe class imbalance, accuracy alone is not sufficient for evaluating model performance. Recall and F1-score should also be considered when assessing the model's ability to identify stroke cases.

---

## 📈 Top Important Features

| Feature                       | Importance |
| ----------------------------- | ---------- |
| Age                           | 0.673      |
| BMI                           | 0.225      |
| Average Glucose Level         | 0.078      |
| Hypertension                  | 0.009      |
| Smoking Status (Never Smoked) | 0.009      |

### Key Insight

Age, BMI, and Average Glucose Level are the strongest predictors of stroke risk in this dataset.

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)

---

## 🚀 Future Improvements

* Threshold Tuning
* Cost-Sensitive Learning
* Ensemble Comparison (Random Forest, Gradient Boosting, XGBoost)
* Advanced Feature Engineering
* Model Deployment using Flask or Streamlit

---

## 📜 Conclusion

This project demonstrates a complete machine learning workflow for stroke prediction using AdaBoost. The study highlights the importance of addressing class imbalance and evaluating models beyond accuracy. Feature importance analysis revealed that age, BMI, and glucose levels are significant factors influencing stroke risk.
