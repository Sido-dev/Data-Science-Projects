# Melbourne Housing Price Prediction

## Overview

This project focuses on predicting house prices using the Melbourne Housing Dataset. The workflow includes data cleaning, missing value handling, feature engineering, categorical encoding, model training, and evaluation using Linear Regression.

The project demonstrates a complete Machine Learning regression pipeline using Python and Scikit-learn.

---

# Workflow

```text
Load Dataset
    ↓
Data Cleaning
    ↓
Missing Value Handling
    ↓
Feature Engineering
    ↓
Encoding Categorical Features
    ↓
Train-Test Split
    ↓
Model Training
    ↓
Prediction
    ↓
Model Evaluation
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Dataset Information

The dataset contains housing-related information such as:

| Feature | Description |
|---|---|
| Rooms | Number of rooms |
| Price | House price |
| Bathroom | Number of bathrooms |
| Car | Parking spaces |
| Landsize | Land size |
| Type | Property type |
| Method | Selling method |
| Regionname | Property region |

---

# Data Preprocessing

## Missing Value Handling

- Dropped highly missing columns
- Median imputation for numerical columns
- Mode imputation for categorical columns
- Random imputation for selected features

## Encoding Techniques

- One Hot Encoding for low-cardinality features
- Frequency Encoding for high-cardinality features

## Date Feature Extraction

Extracted:
- Year
- Month
- Day

from the `Date` column.

---

# Model Building

## Algorithm Used

- Linear Regression

## Train-Test Split

- 75% Training Data
- 25% Testing Data

---

# Model Evaluation

Evaluation metrics used:

- RMSE (Root Mean Squared Error)
- R² Score

## Results

| Metric | Value |
|---|---|
| RMSE | ~421,788 |
| R² Score | ~0.58 |

---

# Key Learnings

- Handling missing values professionally
- Choosing suitable encoding techniques
- Feature engineering from dates
- Building regression models
- Evaluating model performance

---

# Future Improvements

Possible future enhancements:

- Random Forest Regressor
- XGBoost Regressor
- Hyperparameter tuning
- Outlier handling
- Advanced feature engineering

---

# Conclusion

This project demonstrates an end-to-end Machine Learning workflow for housing price prediction using the Melbourne Housing Dataset.

The Multiple Linear Regression model achieved moderate performance and provides a strong baseline for future improvements using advanced regression algorithms and feature engineering techniques.