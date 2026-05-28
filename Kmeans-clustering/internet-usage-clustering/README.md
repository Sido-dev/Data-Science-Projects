# 🌐 Internet Usage Analysis using K-Means Clustering

## 📌 Overview
This project analyzes internet usage trends from **2000 to 2022** using the **K-Means Clustering** algorithm. The goal is to identify similar internet usage growth patterns through unsupervised learning.

---

## 📖 About K-Means Clustering
K-Means is an **Unsupervised Machine Learning Algorithm** used to group similar data points into clusters based on feature similarity.

---

## ⚙️ Workflow

```text
Import Libraries
        ↓
Load Dataset
        ↓
Data Understanding
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
Find Optimal Clusters (Elbow Method)
        ↓
Train K-Means Model
        ↓
Assign Clusters
        ↓
Cluster Visualization
        ↓
Silhouette Score Evaluation
        ↓
Insights & Conclusion
```

---

## 📚 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

---

## 📊 Visualizations

### Correlation Heatmap

```python
sns.heatmap(df.corr(), annot=True)
plt.show()
```

### Cluster Visualization

```python
sns.scatterplot(x=df.index, y=df["2022"], hue=df["cluster"])

plt.show()
```

---

## 📈 Model Evaluation

```python
silhouette_score(scaled_data, df["cluster"])
```

---

## 🔍 Insights
- Internet usage gradually increased from 2000 to 2022.
- K-Means successfully grouped similar internet usage patterns.
- Different clusters represented different growth trends.

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📁 Project Structure

```text
Internet-Usage-KMeans/
│
├── internet_usage.csv
├── internet_usage.ipynb
├── README.md
```

---

## ⭐ Conclusion
K-Means Clustering effectively identified meaningful internet usage patterns and growth trends from 2000 to 2022.

⭐ If you found this project useful, consider giving it a star.