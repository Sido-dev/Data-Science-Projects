# Regularization in Machine Learning

## Overview
Regularization is a technique used in Machine Learning to reduce overfitting and improve a model's ability to generalize to unseen data. It works by adding a penalty term to the loss function, discouraging overly complex models.

---

## Why Regularization?

When a model becomes too complex, it may memorize training data instead of learning patterns. This leads to:

- High training accuracy
- Poor testing accuracy
- Overfitting

Regularization helps create simpler and more robust models.

---

## Types of Regularization

### 1. L1 Regularization (Lasso)

Adds the absolute value of coefficients as a penalty.

**Formula:**

Loss Function = Original Loss + λ × Σ|w|

**Characteristics:**
- Performs feature selection
- Can reduce some coefficients to zero
- Produces sparse models

**Use Case:**
- When dealing with high-dimensional datasets
- When feature selection is required

---

### 2. L2 Regularization (Ridge)

Adds the squared value of coefficients as a penalty.

**Formula:**

Loss Function = Original Loss + λ × Σw²

**Characteristics:**
- Shrinks coefficients toward zero
- Does not eliminate features completely
- Reduces model complexity

**Use Case:**
- When all features are important
- To reduce multicollinearity

---

### 3. Elastic Net

Combines both L1 and L2 Regularization.

**Formula:**

Loss Function = Original Loss + λ₁ × Σ|w| + λ₂ × Σw²

**Characteristics:**
- Feature selection + coefficient shrinkage
- Balances advantages of Lasso and Ridge

**Use Case:**
- Large datasets with correlated features

---

## Effect of Lambda (λ)

Lambda controls the strength of regularization.

| Lambda Value | Effect |
|-------------|---------|
| λ = 0 | No regularization |
| Small λ | Slight penalty |
| Large λ | Strong penalty |
| Very Large λ | Underfitting may occur |

---

## Advantages

- Reduces overfitting
- Improves model generalization
- Handles multicollinearity
- Prevents extremely large coefficients
- Improves model stability

---

## Disadvantages

- May introduce underfitting if penalty is too strong
- Requires hyperparameter tuning
- Can increase training complexity

---

## Comparison

| Feature | Lasso (L1) | Ridge (L2) | Elastic Net |
|----------|------------|------------|-------------|
| Feature Selection | ✅ | ❌ | ✅ |
| Removes Features | ✅ | ❌ | ✅ |
| Handles Multicollinearity | Moderate | Excellent | Excellent |
| Sparse Model | ✅ | ❌ | ✅ |

---

## Applications

- Linear Regression
- Logistic Regression
- Polynomial Regression
- High-Dimensional Data Analysis
- Feature Selection Problems

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Conclusion

Regularization is an essential technique for building reliable Machine Learning models. L1 (Lasso) is useful for feature selection, L2 (Ridge) helps reduce coefficient magnitude, and Elastic Net combines the strengths of both methods. Choosing the appropriate regularization technique can significantly improve model performance and generalization.