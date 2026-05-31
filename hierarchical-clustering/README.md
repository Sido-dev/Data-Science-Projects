# Hierarchical Clustering

## 📌 Overview
Hierarchical Clustering is an unsupervised machine learning algorithm used to group similar data points into clusters. It builds a hierarchy of clusters and represents them using a tree-like structure called a **Dendrogram**. Unlike K-Means, it does not require specifying the number of clusters at the beginning.

---

## 📖 What is Hierarchical Clustering? (Theory)

Hierarchical Clustering creates clusters by either:

### 1. Agglomerative Clustering (Bottom-Up)
- Each data point starts as an individual cluster.
- The closest clusters are merged repeatedly.
- The process continues until all points belong to a single cluster.

### 2. Divisive Clustering (Top-Down)
- All data points start in one cluster.
- Clusters are split recursively until each point forms its own cluster.

**Agglomerative Clustering** is the most commonly used approach.

---

## 🔗 Linkage Methods

Linkage determines how the distance between clusters is calculated.

### Single Linkage
Distance between the closest points of two clusters.

### Complete Linkage
Distance between the farthest points of two clusters.

### Average Linkage
Average distance between all points of two clusters.

### Ward Linkage
Minimizes the variance within clusters and often produces compact clusters.

---

## ✨ Features of Hierarchical Clustering

- No need to specify the number of clusters initially.
- Produces a Dendrogram for visualization.
- Easy to understand and interpret.
- Works well with small and medium-sized datasets.
- Can identify nested clusters.
- Supports different linkage methods.

### Limitations

- Computationally expensive for large datasets.
- Sensitive to noise and outliers.
- Difficult to reassign points once clusters are merged.
- Performance decreases with very large datasets.

---

## ⚙️ Workflow of the Model

### Step 1: Data Collection
Collect the dataset for clustering.

### Step 2: Data Preprocessing
- Handle missing values
- Remove duplicates
- Clean inconsistent data

### Step 3: Feature Selection
Choose relevant features for clustering.

### Step 4: Feature Scaling
Standardize or normalize numerical features.

### Step 5: Calculate Distance Matrix
Compute pairwise distances between data points.

### Step 6: Choose Linkage Method
Select:
- Single
- Complete
- Average
- Ward

### Step 7: Build Dendrogram
Merge clusters based on the chosen linkage method.

### Step 8: Determine Optimal Number of Clusters
Cut the dendrogram at the desired height.

### Step 9: Form Clusters
Assign cluster labels to observations.

### Step 10: Evaluate Results
Analyze:
- Cluster distribution
- Cluster separation
- Silhouette Score (if applicable)

### Step 11: Visualization
Visualize clusters and dendrogram.

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
Distance Matrix Calculation
       ↓
Choose Linkage Method
       ↓
Build Dendrogram
       ↓
Determine Optimal Clusters
       ↓
Assign Cluster Labels
       ↓
Evaluate Results
       ↓
Visualization
       ↓
Conclusion
```

---

## 📊 Result

After applying Hierarchical Clustering:

- Similar data points are grouped into clusters.
- A Dendrogram is generated to visualize cluster relationships.
- The optimal number of clusters can be selected by cutting the dendrogram.
- Each observation receives a cluster label.

### Example Output

| Data Point | Cluster Label |
|------------|--------------|
| Point 1 | 0 |
| Point 2 | 0 |
| Point 3 | 1 |
| Point 4 | 1 |
| Point 5 | 2 |

---

## 🌳 Dendrogram

A Dendrogram is a tree-like diagram that shows how clusters are merged during the clustering process.

### Purpose of Dendrogram

- Visualize cluster formation.
- Determine the optimal number of clusters.
- Understand relationships among data points.

---

## 🎯 Conclusion

Hierarchical Clustering is a powerful clustering technique that builds a hierarchy of clusters and provides a visual representation through a dendrogram. It is especially useful when the number of clusters is unknown and when understanding relationships between observations is important. Ward linkage is commonly preferred because it tends to create compact and well-separated clusters.

### Key Takeaways

- **Algorithm Type:** Unsupervised Learning
- **Technique:** Hierarchical Clustering
- **Approaches:** Agglomerative & Divisive
- **Popular Linkage:** Ward Linkage
- **Output:** Dendrogram + Cluster Labels
- **Best Use Case:** Discovering natural groupings and relationships in data