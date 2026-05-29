# Credit Card Customer Segmentation using DBSCAN and PCA

## 📌 Project Overview

This project performs Customer Segmentation on credit card customer data using the DBSCAN Clustering Algorithm combined with PCA (Principal Component Analysis).

The objective of this project is to:

- Identify customer spending behavior patterns
- Detect anomalous/outlier customers
- Group customers based on financial activity
- Visualize customer clusters effectively

DBSCAN was chosen because it can automatically detect clusters and identify noise/outliers without requiring a predefined number of clusters.

---

# 📂 Dataset Information

Dataset Name:

```text
credit_card_dataset.csv
```

The dataset contains customer financial behavior attributes such as:

| Feature | Description |
|---|---|
| BALANCE | Remaining account balance |
| PURCHASES | Total purchase amount |
| ONEOFF_PURCHASES | One-time purchases |
| INSTALLMENTS_PURCHASES | Installment purchases |
| CASH_ADVANCE | Cash advance amount |
| PURCHASES_FREQUENCY | Purchase frequency |
| CASH_ADVANCE_FREQUENCY | Cash advance frequency |
| PURCHASES_TRX | Number of purchase transactions |
| CREDIT_LIMIT | Credit card limit |
| PAYMENTS | Total payments |
| MINIMUM_PAYMENTS | Minimum payment amount |
| PRC_FULL_PAYMENT | Percentage of full payment |
| TENURE | Credit card tenure |

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

# 📌 Project Flow

1. Data Collection  
2. Data Understanding  
3. Data Cleaning  
4. Missing Value Handling  
5. Exploratory Data Analysis (EDA)  
6. Feature Selection  
7. Feature Scaling using StandardScaler  
8. Dimensionality Reduction using PCA  
9. DBSCAN Model Building  
10. Hyperparameter Tuning (`eps`, `min_samples`)  
11. Cluster Prediction  
12. Cluster Visualization  
13. Silhouette Score Evaluation  
14. Outlier Detection  
15. Customer Segmentation Analysis  
16. Insights & Conclusion  

---

# ⚠️ Limitations

- Sensitive to `eps` parameter
- Difficult with varying density datasets
- Large dominant clusters may occur in financial datasets

---

# 📌 Conclusion

This project demonstrates how DBSCAN and PCA can be combined for effective customer segmentation and anomaly detection in financial datasets.

The model successfully identified:

- Customer spending behavior patterns
- Dense customer groups
- Outlier/anomalous customers

making it useful for customer analytics and financial behavior analysis.

---
