# Sampling Techniques in Machine Learning

## 📌 Project Overview

This project demonstrates various **Sampling Techniques** used in Machine Learning to handle imbalanced datasets. Sampling helps create a balanced distribution of classes, leading to improved model performance and more reliable predictions.

---

## 🎯 Objective

To explore and compare different sampling techniques used for balancing datasets and improving classification results.

---

## 🎯 Where Do We Use Sampling?

| Technique | Best Used When | Example Dataset |
|------------|---------------|----------------|
| Random Over Sampling | Minority class has very few samples and data duplication is acceptable. | Credit Card Fraud Detection |
| Random Under Sampling | Majority class significantly outweighs minority class. | Customer Churn Prediction |
| SMOTE | Generating synthetic samples for minority classes. | Heart Disease Prediction |
| ADASYN | Minority class samples are difficult to learn and require adaptive generation. | Loan Default Prediction |
| NearMiss | Selecting informative majority-class samples. | Fraud Detection |
| SMOTEENN | Combining oversampling and noise removal. | Medical Diagnosis |
| SMOTETomek | Balancing classes while removing overlapping samples. | Customer Segmentation |

---

## 🛠 Techniques Used

### 1. Random Over Sampling
Increases the minority class size by randomly duplicating existing samples.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)
```

### 2. Random Under Sampling
Reduces the majority class size by randomly removing samples.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
```

### 3. SMOTE (Synthetic Minority Over-sampling Technique)
Creates synthetic minority-class samples instead of duplicating data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### 4. ADASYN
Generates synthetic samples adaptively based on data complexity.

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X, y)
```

### 5. NearMiss
Undersamples the majority class by selecting the closest samples to minority instances.

```python
from imblearn.under_sampling import NearMiss

nm = NearMiss()
X_resampled, y_resampled = nm.fit_resample(X, y)
```

### 6. SMOTEENN
Combines SMOTE with Edited Nearest Neighbours for oversampling and noise removal.

```python
from imblearn.combine import SMOTEENN

smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X, y)
```

### 7. SMOTETomek
Combines SMOTE with Tomek Links to remove overlapping observations.

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)
```

---

## 📂 Workflow

1. Load Dataset
2. Analyze Class Distribution
3. Apply Sampling Techniques
4. Train Machine Learning Models
5. Compare Performance Metrics
6. Evaluate Class Balance Improvements

---

## 📊 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (imblearn)

---

## 📈 Benefits of Sampling

- Handles class imbalance
- Improves model performance
- Reduces prediction bias
- Increases minority class representation
- Enhances recall and F1-score

---

## 🚀 Conclusion

Sampling techniques are essential for handling imbalanced datasets. Oversampling methods increase minority class representation, undersampling methods reduce majority class dominance, and hybrid methods combine both approaches to achieve better classification performance.

---

## 👨‍💻 Author

**Sudhanshu Narayane**  
Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer