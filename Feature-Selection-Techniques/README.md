# Feature Selection Techniques in Machine Learning

## 📌 Project Overview
This project demonstrates multiple feature selection techniques used in Machine Learning to identify the most relevant features for predictive modeling. Feature selection helps improve model performance, reduce overfitting, and decrease computational complexity.

---

## 🎯 Objective
Compare different feature selection methods and identify the most important features from a dataset.

## 🎯 Where Do We Use Feature Selection?

Feature selection is commonly used before training Machine Learning models to select the most relevant features and remove unnecessary ones.

### Use Cases

- Improving model accuracy by keeping only important features.
- Reducing overfitting caused by irrelevant variables.
- Decreasing training time on large datasets.
- Simplifying model interpretation and explainability.
- Handling high-dimensional datasets with many features.

### When to Use Each Technique

| Technique | Best Used When | Example Dataset |
|------------|---------------|----------------|
| Chi-Square (Chi²) | Features are categorical or non-negative and the target is categorical. | Titanic Survival Prediction |
| Correlation | Finding linear relationships between numerical features and the target. | House Price Prediction |
| Decision Tree Importance | Identifying important features for tree-based models. | Heart Disease Prediction |
| RFE | Selecting the optimal subset of features using a machine learning model. | Customer Churn Prediction |
| Slope Method (m-Slope) | Measuring the linear influence of a feature on the target variable. | Salary Prediction, Sales Forecasting |

### Real-World Applications

- **Healthcare:** Selecting key patient attributes for disease prediction.
- **Finance:** Identifying factors affecting loan approval or fraud detection.
- **Marketing:** Finding customer characteristics that influence purchases.
- **E-commerce:** Selecting features for recommendation systems.
- **Predictive Analytics:** Reducing dimensionality before model training.

---

## 🛠 Techniques Used

### 1. Chi-Square (Chi²) Test
Selects features based on their statistical relationship with the target variable.

```python
from sklearn.feature_selection import SelectKBest, chi2

selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)
```

### 2. Correlation-Based Selection
Selects features with strong correlation to the target variable.

```python
import pandas as pd

corr_matrix = df.corr(numeric_only=True)
target_corr = corr_matrix['target'].abs().sort_values(ascending=False)
print(target_corr)
```

### 3. Decision Tree Feature Importance
Uses a Decision Tree model to rank features based on importance.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

importance = model.feature_importances_
print(importance)
```

### 4. Recursive Feature Elimination (RFE)
Recursively removes less important features until the desired number remains.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)

X_rfe = rfe.fit_transform(X, y)
```

### 5. Slope Method (m-Slope)
Calculates the slope between a feature and target to measure linear influence.

```python
from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(X['feature'], y)

print("Slope:", slope)
```

---

## 📂 Workflow

1. Load Dataset
2. Data Cleaning & Preprocessing
3. Apply Feature Selection Methods:
   - Chi-Square Test
   - Correlation Analysis
   - Decision Tree Importance
   - Recursive Feature Elimination (RFE)
   - Slope Method
4. Compare Selected Features
5. Visualize Results
6. Choose Best Features for Model Training

---

## 📊 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

## 📈 Benefits of Feature Selection

- Reduces Overfitting
- Improves Model Accuracy
- Faster Training
- Better Interpretability
- Removes Redundant Features

---

## 🚀 Conclusion

Different feature selection techniques may select different features because each method evaluates feature importance differently. Comparing multiple approaches helps identify the most influential features and build more efficient machine learning models.

---

## 👨‍💻 Author

**Sudhanshu Narayane**  
Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer