# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## 📌 Overview
DBSCAN is a density-based clustering algorithm used in Unsupervised Machine Learning. It groups data points that are closely packed together and identifies points in low-density regions as outliers (noise). Unlike K-Means, DBSCAN does not require specifying the number of clusters beforehand.

---

## 📖 What is DBSCAN? (Theory)

DBSCAN forms clusters based on the density of data points in a region.

It uses two important parameters:

- **Epsilon (ε):** Maximum distance between two points to be considered neighbors.
- **MinPts:** Minimum number of neighboring points required to form a dense region.

### Types of Points

### 1. Core Point
A point having at least **MinPts** points within its ε-neighborhood.

### 2. Border Point
A point that has fewer than **MinPts** neighbors but lies within the neighborhood of a core point.

### 3. Noise Point (Outlier)
A point that does not belong to any cluster and is not reachable from a core point.

---

## ✨ Features of DBSCAN

- No need to specify the number of clusters.
- Detects clusters of arbitrary shapes.
- Identifies noise and outliers automatically.
- Works well with spatial and geographical data.
- Robust to outliers.
- Effective for datasets with irregular cluster structures.

### Limitations

- Sensitive to ε (epsilon) and MinPts values.
- Struggles when clusters have varying densities.
- Less effective in high-dimensional datasets.

---

## ⚙️ Workflow of the Model

### Step 1: Data Collection
Collect the dataset for clustering.

### Step 2: Data Preprocessing
- Handle missing values
- Remove duplicates
- Clean inconsistent data

### Step 3: Feature Selection
Select relevant features for clustering.

### Step 4: Feature Scaling
Normalize or standardize numerical features.

### Step 5: Determine Optimal ε
Use a K-Distance Graph to select the appropriate epsilon value.

### Step 6: Apply DBSCAN
Train the DBSCAN model using:
- ε (Epsilon)
- MinPts

### Step 7: Generate Clusters
The algorithm groups dense regions into clusters.

### Step 8: Detect Noise
Points that do not belong to any cluster are labeled as outliers.

### Step 9: Evaluate Results
Analyze:
- Number of clusters
- Cluster sizes
- Noise points
- Silhouette Score (if applicable)

### Step 10: Visualization
Visualize clusters using scatter plots.

---

## 🔄 Workflow Diagram

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Feature Scaling
       ↓
Find Optimal ε
       ↓
Apply DBSCAN
       ↓
Generate Clusters
       ↓
Detect Noise Points
       ↓
Evaluate Results
       ↓
Visualization
       ↓
Conclusion
```

---

## 📊 Result

After applying DBSCAN:

- Dense data points are grouped into clusters.
- Outliers are automatically identified as noise.
- The number of clusters is determined by the algorithm.
- Cluster labels are assigned to each observation.
- Noise points are labeled as **-1**.

### Example Output

| Data Point | Cluster Label |
|------------|--------------|
| Point 1 | 0 |
| Point 2 | 0 |
| Point 3 | 1 |
| Point 4 | 1 |
| Point 5 | -1 (Noise) |

---

## 🎯 Conclusion

DBSCAN is a powerful density-based clustering algorithm that can discover clusters of varying shapes without requiring a predefined number of clusters. It is particularly useful for datasets containing noise and outliers. Proper selection of ε and MinPts is crucial for achieving optimal clustering performance.

### Key Takeaways

- **Algorithm Type:** Unsupervised Learning
- **Technique:** Density-Based Clustering
- **Main Parameters:** ε (Epsilon), MinPts
- **Output:** Clusters + Noise (Outliers)
- **Best Use Case:** Datasets with irregular cluster shapes and unknown number of clusters