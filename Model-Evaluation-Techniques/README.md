# Model Evaluation Techniques in Machine Learning

## 📌 Project Overview

This project demonstrates various **Model Evaluation Techniques** used in Machine Learning to assess model performance and compare different algorithms. Proper evaluation helps determine how well a model generalizes to unseen data and supports selecting the best model for a given problem.

---

## 🎯 Objective

To explore and compare different evaluation metrics used for both **Supervised** and **Unsupervised Learning** tasks in Machine Learning.

---

## 📋 Evaluation Techniques Categories

### Supervised Learning

#### Classification Metrics

| Technique | Best Used When | Example Dataset |
|------------|---------------|----------------|
| Accuracy | Balanced classification problems. | Iris Classification |
| Precision | False positives are costly. | Spam Email Detection |
| Recall | False negatives are costly. | Disease Detection |
| F1-Score | Need balance between Precision and Recall. | Fraud Detection |
| ROC-AUC | Comparing classifiers across thresholds. | Credit Risk Prediction |
| Confusion Matrix | Understanding prediction errors. | Customer Churn Prediction |

#### Regression Metrics

| Technique | Best Used When | Example Dataset |
|------------|---------------|----------------|
| MAE | Measuring average prediction error. | House Price Prediction |
| MSE | Penalizing large errors more heavily. | Sales Forecasting |
| RMSE | Error measurement in original target units. | Flight Price Prediction |
| R² Score | Measuring explained variance. | Salary Prediction |
| Adjusted R² | Comparing models with different feature counts. | Real Estate Prediction |
| MAPE | Percentage-based error evaluation. | Demand Forecasting |

---

### Unsupervised Learning

#### Clustering Metrics

| Technique | Best Used When | Example Dataset |
|------------|---------------|----------------|
| Silhouette Score | Evaluating cluster separation and cohesion. | Customer Segmentation |
| Davies-Bouldin Index | Comparing clustering quality. | Market Basket Analysis |
| Calinski-Harabasz Score | Measuring cluster density and separation. | User Behavior Analysis |
| Inertia (WCSS) | Finding optimal K in K-Means. | Customer Grouping |
| Elbow Method | Determining the ideal number of clusters. | Retail Segmentation |

#### Dimensionality Reduction Metrics

| Technique | Best Used When | Example Dataset |
|------------|---------------|----------------|
| Explained Variance Ratio | PCA performance evaluation. | Image Compression |
| Reconstruction Error | Autoencoder evaluation. | Anomaly Detection |

---

## 🛠 Evaluation Techniques

# Supervised Learning

## Classification Metrics

### 1. Accuracy

Measures the proportion of correctly classified instances.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

### 2. Precision

Measures how many predicted positives are actually positive.

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print(precision)
```

### 3. Recall

Measures how many actual positives are correctly identified.

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)
print(recall)
```

### 4. F1-Score

Harmonic mean of Precision and Recall.

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, y_pred)
print(f1)
```

### 5. Confusion Matrix

Shows correct and incorrect predictions.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
```

### 6. ROC-AUC Score

Measures model discrimination ability.

```python
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(roc_auc)
```

---

## Regression Metrics

### 7. Mean Absolute Error (MAE)

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(mae)
```

### 8. Mean Squared Error (MSE)

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(mse)
```

### 9. Root Mean Squared Error (RMSE)

```python
from sklearn.metrics import root_mean_squared_error

rmse = root_mean_squared_error(y_test, y_pred)
print(rmse)
```

### 10. R² Score

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(r2)
```

### 11. Adjusted R²

```python
n = len(y_test)
p = X_test.shape[1]

adj_r2 = 1 - ((1-r2)*(n-1)/(n-p-1))
print(adj_r2)
```

### 12. Mean Absolute Percentage Error (MAPE)

```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred)
print(mape)
```

---

# Unsupervised Learning

## Clustering Metrics

### 13. Silhouette Score

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
print(score)
```

### 14. Davies-Bouldin Index

```python
from sklearn.metrics import davies_bouldin_score

score = davies_bouldin_score(X, labels)
print(score)
```

### 15. Calinski-Harabasz Score

```python
from sklearn.metrics import calinski_harabasz_score

score = calinski_harabasz_score(X, labels)
print(score)
```

### 16. Inertia (WCSS)

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

print(kmeans.inertia_)
```

### 17. Elbow Method

```python
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
```

---

## Dimensionality Reduction Metrics

### 18. Explained Variance Ratio (PCA)

```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)

print(pca.explained_variance_ratio_)
```

### 19. Reconstruction Error

```python
import numpy as np

reconstruction_error = np.mean((X - X_reconstructed) ** 2)
print(reconstruction_error)
```

---

## 📂 Workflow

1. Load Dataset
2. Data Cleaning & Preprocessing
3. Train Machine Learning or Clustering Model
4. Generate Predictions or Clusters
5. Apply Evaluation Metrics
6. Compare Results
7. Select the Best Model
8. Interpret Findings

---

## 📊 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📈 Benefits of Model Evaluation

- Measures model performance objectively
- Helps compare multiple algorithms
- Detects overfitting and underfitting
- Improves model reliability
- Supports data-driven decision making
- Enhances model interpretability

---

## 🚀 Conclusion

Model evaluation is a critical step in the Machine Learning workflow. Different metrics provide different insights into model performance, and choosing the appropriate metric depends on the learning task and business objective. By combining classification, regression, clustering, and dimensionality reduction metrics, practitioners can build more reliable and effective machine learning solutions.

---

## 👨‍💻 Author

**Sudhanshu Narayane**  
Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer