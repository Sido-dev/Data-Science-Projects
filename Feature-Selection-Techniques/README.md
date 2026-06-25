# Feature Selection Techniques in Machine Learning

## 📌 Project Overview

This project demonstrates and compares various **Feature Selection Techniques** used in Machine Learning to identify the most relevant features for predictive modeling. Feature selection helps reduce dimensionality, improve model performance, decrease training time, and enhance model interpretability.

---

## 🎯 Objective

To explore and compare multiple feature selection methods and identify the most important features for building efficient machine learning models.

---

## 🎯 Where Do We Use Feature Selection?

| Technique | Best Used When | Example Dataset |
|------------|---------------|----------------|
| Variance Threshold | Removing features with very low variance. | Customer Demographics |
| Chi-Square (Chi²) | Features are categorical or non-negative and the target is categorical. | Titanic Survival Prediction |
| Correlation Analysis | Finding linear relationships between numerical features and the target. | House Price Prediction |
| ANOVA F-Test | Comparing numerical features against categorical targets. | Student Performance Prediction |
| Mutual Information | Capturing linear and non-linear relationships. | Customer Churn Prediction |
| Decision Tree Importance | Identifying important features for tree-based models. | Heart Disease Prediction |
| Random Forest Importance | Measuring feature importance using ensemble learning. | Credit Risk Analysis |
| XGBoost Importance | Selecting features in gradient boosting models. | Loan Default Prediction |
| Recursive Feature Elimination (RFE) | Selecting the optimal subset of features. | Employee Attrition Prediction |
| RFECV | Automatically determining the optimal number of features. | Sales Prediction |
| Forward Feature Selection | Adding features one by one based on performance improvement. | House Price Prediction |
| Backward Feature Elimination | Removing less important features iteratively. | Medical Diagnosis |
| Permutation Importance | Measuring performance drop when a feature is shuffled. | Fraud Detection |
| SHAP | Explaining feature contributions to model predictions. | Customer Segmentation |
| Slope Method (m) *(Optional)* | Measuring the linear influence of a feature on the target. | Sales Forecasting |

---

## 🛠 Feature Selection Techniques

### Filter Methods

#### 1. Variance Threshold
Removes features with low variance.

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

#### 2. Chi-Square (Chi²)
Measures dependency between features and the target.

```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=chi2, k=5)
X_selected = selector.fit_transform(X, y)
```

#### 3. Correlation Analysis
Selects features based on correlation strength.

```python
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix['target'].sort_values(ascending=False))
```

#### 4. ANOVA F-Test
Evaluates numerical features against categorical targets.

```python
from sklearn.feature_selection import f_classif

scores, p_values = f_classif(X, y)
```

#### 5. Mutual Information
Measures linear and non-linear dependency.

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
```

---

### Wrapper Methods

#### 6. Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)

X_selected = rfe.fit_transform(X, y)
```

#### 7. RFECV

```python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    cv=5
)

X_selected = rfecv.fit_transform(X, y)
```

#### 8. Forward Feature Selection

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

sfs = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000),
    direction='forward'
)

X_selected = sfs.fit_transform(X, y)
```

#### 9. Backward Feature Elimination

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

sfs = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000),
    direction='backward'
)

X_selected = sfs.fit_transform(X, y)
```

---

### Embedded Methods

#### 10. Decision Tree Feature Importance

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)

print(model.feature_importances_)
```

#### 11. Random Forest Feature Importance

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)

print(model.feature_importances_)
```

#### 12. XGBoost Feature Importance

```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X, y)

print(model.feature_importances_)
```

---

### Explainability Methods

#### 13. Permutation Importance

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,
    X_test,
    y_test
)

print(result.importances_mean)
```

#### 14. SHAP

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)
```

---

### Statistical Method (Optional)

#### 15. Slope Method (m)

Measures the rate of change between a feature and the target variable.

```python
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(
    X['feature'],
    y
)

print("Slope:", slope)
```

---

## 📋 Feature Selection Categories

| Category | Techniques |
|-----------|-----------|
| Filter Methods | Variance Threshold, Chi-Square (Chi²), Correlation Analysis, ANOVA F-Test, Mutual Information |
| Wrapper Methods | RFE, RFECV, Forward Feature Selection, Backward Feature Elimination |
| Embedded Methods | Decision Tree Importance, Random Forest Importance, XGBoost Importance |
| Explainability Methods | Permutation Importance, SHAP |
| Statistical Methods | Slope Method (m), Linear Trend Analysis |

---

## 📂 Workflow

1. Load Dataset
2. Data Cleaning & Preprocessing
3. Feature Encoding and Scaling
4. Apply Feature Selection Techniques
5. Compare Selected Features
6. Visualize Feature Importance
7. Train Machine Learning Models
8. Evaluate Model Performance

---

## 📊 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- SciPy

---

## 📈 Benefits of Feature Selection

- Reduces Overfitting
- Improves Model Accuracy
- Faster Training
- Reduces Computational Cost
- Removes Redundant Features
- Enhances Interpretability

---

## 🚀 Conclusion

Feature selection is a critical preprocessing step in Machine Learning. Different techniques evaluate feature importance from different perspectives, including statistical relationships, model performance, embedded importance scores, and explainability methods. Combining multiple approaches often results in better feature selection and more robust machine learning models.

---

## 👨‍💻 Author

**Sudhanshu Narayane**  
Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer