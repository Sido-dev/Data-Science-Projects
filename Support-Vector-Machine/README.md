# Support Vector Machine (SVM)

## Theory

Support Vector Machine (SVM) is a **supervised machine learning algorithm** used for **classification** and **regression** problems.

The goal of SVM is to find the **optimal hyperplane** that separates classes while maximizing the distance (margin) between the nearest data points.

The nearest points to the boundary are called **Support Vectors** because they determine the position of the decision boundary.

---

## Core Concepts

### Hyperplane
A decision boundary used to separate classes.

Example:
- 2D → Line
- 3D → Plane
- Higher Dimensions → Hyperplane

### Margin
Distance between the hyperplane and closest data points.

- Larger Margin → Better Generalization
- Smaller Margin → Higher chance of overfitting

### Support Vectors
Data points nearest to the decision boundary.

### Kernel Trick
Used when data is not linearly separable.

Common kernels:
- Linear
- Polynomial
- RBF (Radial Basis Function)
- Sigmoid

### Hyperparameters

#### C Parameter
Controls tradeoff between margin and classification error.

- Small C → Larger margin
- Large C → Less classification error

#### Gamma
Controls influence of individual data points.

- Low Gamma → Smooth boundary
- High Gamma → Complex boundary

---

## Libraries Required

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.svm import (
    SVC,
    SVR
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
```

---

## Workflow

### 1. Import Libraries

Load all required packages.

---

### 2. Load Data

```python
df = pd.read_csv("dataset.csv")

df.head()
```

---

### 3. Data Understanding

```python
df.shape

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()
```

Tasks:
- Check rows and columns
- Missing values
- Data types
- Statistical summary

---

### 4. Exploratory Data Analysis (EDA)

```python
sns.pairplot(df)

plt.show()
```

Additional Analysis:

```python
sns.heatmap(
    df.corr(),
    annot=True
)

plt.show()
```

---

### 5. Data Preprocessing

Separate features and target:

```python
X = df.drop(
    "target",
    axis=1
)

y = df["target"]
```

Feature Scaling:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

### 6. Train Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

---

### 7. Model Training

```python
model = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

model.fit(
    X_train,
    y_train
)
```

---

### 8. Hyperparameter Tuning

```python
params = {
    "kernel": [
        "linear",
        "rbf"
    ],

    "C": [
        0.1,
        1,
        10
    ],

    "gamma": [
        0.1,
        0.01
    ]
}

grid = GridSearchCV(
    model,
    params,
    cv=5
)

grid.fit(
    X_train,
    y_train
)

grid.best_params_
```

---

### 9. Prediction

```python
y_pred = model.predict(
    X_test
)
```

---

### 10. Model Evaluation

Accuracy:

```python
accuracy_score(
    y_test,
    y_pred
)
```

Classification Report:

```python
classification_report(
    y_test,
    y_pred
)
```

Confusion Matrix:

```python
sns.heatmap(
    confusion_matrix(
        y_test,
        y_pred
    ),
    annot=True
)

plt.show()
```

---

## Evaluation Metrics

### Classification

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Regression (SVR)

- MAE
- MSE
- RMSE
- R² Score

---

## Results and Observations

- SVM performs well for high-dimensional datasets.
- Feature scaling significantly improves performance.
- RBF kernel generally works well for non-linear data.
- Hyperparameter tuning improves generalization.
- Large C may cause overfitting.

---

## Advantages

- Effective for high-dimensional data
- Works for classification and regression
- Strong generalization capability
- Supports non-linear boundaries

---

## Limitations

- Slow for large datasets
- Sensitive to parameter tuning
- Less interpretable than simple models

---

## Future Improvements

- Compare multiple kernels
- Apply RandomizedSearchCV
- Perform Feature Selection
- Use PCA before SVM
- Deploy using Flask or Streamlit

---

## Conclusion

Support Vector Machine (SVM) builds an optimal decision boundary using support vectors and margin maximization. Proper preprocessing, scaling, kernel selection, and hyperparameter tuning help achieve strong predictive performance.

---

## Author

Sudhanshu Narayane 
GitHub: Sido-dev