# 🌲 Random Forest

## 📌 Overview
Random Forest is a powerful ensemble machine learning algorithm that combines multiple Decision Trees to improve prediction accuracy and reduce overfitting.

It can be used for:
- Classification Problems
- Regression Problems

---

# 📖 Theory

Random Forest works on the principle of **Bagging (Bootstrap Aggregating)**.

### Step 1: Bootstrap Sampling
Random subsets of data are created from the original dataset with replacement.

### Step 2: Build Multiple Decision Trees
A Decision Tree is trained on each random subset.

### Step 3: Random Feature Selection
Instead of considering all features, each tree uses a random subset of features while splitting nodes.

### Step 4: Aggregate Predictions
- **Classification:** Majority Voting
- **Regression:** Average of Predictions

---

# 🔑 Key Features

- Ensemble Learning Algorithm
- Reduces Overfitting compared to a single Decision Tree
- Handles Large Datasets efficiently
- Works with Numerical and Categorical Data
- Provides Feature Importance Scores
- Robust to Noise and Outliers

---

# ⚙️ Important Hyperparameters

| Parameter | Description |
|------------|------------|
| n_estimators | Number of trees in the forest |
| max_depth | Maximum depth of each tree |
| min_samples_split | Minimum samples required to split a node |
| min_samples_leaf | Minimum samples required at a leaf node |
| max_features | Number of features considered at each split |
| random_state | Ensures reproducibility |

---

# 📊 Advantages

✅ High Accuracy

✅ Reduces Overfitting

✅ Handles Missing Values Better

✅ Works Well on Large Datasets

✅ Provides Feature Importance

---

# ❌ Disadvantages

❌ Training can be slower than a single Decision Tree

❌ Requires more memory

❌ Less interpretable than a single tree

---

# 🧠 How Random Forest Reduces Overfitting

Decision Trees tend to memorize training data, causing overfitting.

Random Forest:
1. Creates multiple trees using different data samples.
2. Uses random feature selection.
3. Combines predictions from all trees.

This reduces variance and improves generalization.

---

# 📈 Feature Importance

Random Forest can measure the importance of each feature based on how much it contributes to reducing impurity across all trees.

Example:
```
Feature A → 45%
Feature B → 30%
Feature C → 15%
Feature D → 10%
```

---

# 🛠️ Model Training (Scikit-Learn)

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

### For Classification

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# 📏 Evaluation Metrics

### Regression
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### Classification
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 🚀 Applications

- Flight Price Prediction
- House Price Prediction
- Customer Churn Prediction
- Fraud Detection
- Medical Diagnosis
- Stock Market Analysis
- Recommendation Systems

---

# 📝 Conclusion

Random Forest is one of the most reliable machine learning algorithms due to its ability to combine multiple Decision Trees and produce accurate, stable, and robust predictions while minimizing overfitting.