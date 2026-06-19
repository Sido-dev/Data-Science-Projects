# Adult Income Prediction (SVM)

## Project Overview

This project focuses on predicting whether an individual's annual income exceeds $50,000 using demographic and employment-related information from the Adult Census Income dataset. The classification task is performed using Support Vector Machine (SVM) models with both Linear and RBF kernels.

The project includes data preprocessing, exploratory data analysis (EDA), feature scaling, model training, hyperparameter tuning, and performance evaluation.

---

## Dataset Information

**Dataset:** Adult Census Income Dataset

The dataset was collected from the **UCI Machine Learning Repository** and contains census-based demographic and employment information used for income classification tasks.  

The dataset contains demographic and employment information collected from census data. The objective is to classify individuals into one of two income categories:

* **<=50K**
* **>50K**

### Features

* Age
* Workclass
* Final Weight (fnlwgt)
* Education Number
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain
* Capital Loss
* Hours per Week
* Native Country

### Target Variable

* **Income**

  * 0 → <=50K
  * 1 → >50K

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Project Workflow

### 1. Data Understanding

* Dataset overview
* Feature inspection
* Data type analysis
* Statistical summary

### 2. Data Preprocessing

* Handling missing values (`?`)
* Feature selection
* Encoding categorical variables
* Target variable encoding
* Removing redundant features

### 3. Exploratory Data Analysis (EDA)

* Income distribution analysis
* Age vs Income analysis
* Education analysis
* Occupation analysis

### 4. Feature Scaling

* StandardScaler applied to numerical features

### 5. Model Building

#### Linear SVM

* Support Vector Classifier with Linear Kernel

#### RBF SVM

* Support Vector Classifier with Radial Basis Function Kernel

### 6. Hyperparameter Tuning

GridSearchCV was used to identify the optimal model parameters.

**Best Parameters**

```python
{
    'C': 10,
    'gamma': 0.01,
    'kernel': 'rbf'
}
```

### 7. Model Evaluation

Evaluation metrics used:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Classification Report

---

## Results

| Model         | Description                          |
| ------------- | ------------------------------------ |
| Linear SVM    | Baseline linear classification model |
| RBF SVM       | Non-linear kernel model              |
| Tuned RBF SVM | Optimized model using GridSearchCV   |

The tuned RBF SVM achieved the best performance among all evaluated models.

---

## Project Structure

```text
adult-income-prediction-svm/
│
├── data/
│   ├── adult.data
│   ├── adult.test
│   └── adult.names
│
├── notebooks/
│   └── svm_income_prediction.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Conclusion

This project demonstrates the application of Support Vector Machines for binary classification on real-world census data. After preprocessing, encoding categorical variables, scaling features, and tuning hyperparameters, the optimized RBF Kernel SVM achieved the best classification performance for predicting income categories.
