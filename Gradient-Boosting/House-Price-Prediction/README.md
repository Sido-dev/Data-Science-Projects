# 🏠 House Price Prediction using Gradient Boosting

## 📌 Overview

This project demonstrates an end-to-end Machine Learning workflow for predicting residential house prices using the **Gradient Boosting Regressor** algorithm. The model is trained on the Ames Housing dataset and optimized using hyperparameter tuning to improve predictive performance.

The objective is to accurately estimate house sale prices based on various property characteristics such as location, lot size, building quality, basement features, garage information, and more.

---

## 📂 Dataset

**Source:** Kaggle House Prices Competition

Dataset Link:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The dataset contains detailed information about residential properties in Ames, Iowa.

### Dataset Files

* `train.csv` – Training data with target variable (`SalePrice`)
* `test.csv` – Test data for generating predictions
* `data_description.txt` – Feature descriptions

---

## 🎯 Problem Statement

Predict the final sale price of residential homes using property-related features and machine learning techniques.

This is a **Regression Problem** where:

* Input: Property Features
* Output: SalePrice

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib
* Jupyter Notebook

---

## 📊 Workflow

### 1. Data Loading

* Imported training and test datasets.
* Performed initial inspection of features and data types.

### 2. Exploratory Data Analysis (EDA)

* Analyzed feature distributions.
* Identified missing values.
* Examined categorical and numerical variables.

### 3. Data Preprocessing

* Missing value imputation.
* One-Hot Encoding for categorical features.
* Numerical feature processing using Scikit-Learn pipelines.

### 4. Model Building

Implemented:

* Gradient Boosting Regressor

### 5. Hyperparameter Tuning

Optimized model performance using RandomizedSearchCV with cross-validation.

Parameters tuned:

* n_estimators
* learning_rate
* max_depth
* subsample
* min_samples_split
* min_samples_leaf

### 6. Model Evaluation

Evaluation Metrics:

* R² Score
* Root Mean Squared Error (RMSE)

### 7. Prediction & Submission

Generated predictions on the test dataset and created a Kaggle-compatible submission file.

---

## 🚀 Model Performance

### Kaggle Public Score

**Score: 0.13327**

The model achieved a Kaggle leaderboard score of **0.13327** on the House Prices: Advanced Regression Techniques competition.

---

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── train.csv
├── test.csv
├── Gradient_Boosting_Project.ipynb
├── gradient_boosting_model.pkl
├── submission_final.csv
├── requirements.txt
└── README.md
```

---

## 💾 Model Saving

The trained model is saved using Joblib:

```python
joblib.dump(best_model, "gradient_boosting_model.pkl")
```

---

## 📈 Feature Engineering & Preprocessing

* Missing Value Imputation
* One-Hot Encoding
* Pipeline-Based Transformation
* Automated Test Data Processing
* Feature Alignment through Scikit-Learn Pipeline

---

## 🔮 Future Improvements

* XGBoost
* LightGBM
* CatBoost
* Ensemble Stacking
* Feature Engineering
* Log Transformation of Target Variable
* Advanced Cross Validation Strategies

---

## 📚 Key Learning Outcomes

* End-to-End Regression Workflow
* Scikit-Learn Pipelines
* Hyperparameter Optimization
* Model Evaluation Techniques
* Kaggle Competition Submission Process
* Production-Ready Machine Learning Practices

---

## 👨‍💻 Author

**Sudhanshu Narayane**

GitHub: https://github.com/sido-dev

LinkedIn: https://www.linkedin.com/in/sid101110

---

⭐ If you found this project useful, consider giving it a star on GitHub.
