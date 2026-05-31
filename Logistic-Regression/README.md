# Logistic Regression

## 📌 Overview

Logistic Regression is a supervised machine learning algorithm used for **classification problems**. It predicts the probability that an observation belongs to a particular class and then assigns it to a category.

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts **categorical outcomes** such as:

- Spam or Not Spam
- Disease or No Disease
- Pass or Fail
- Fraud or Not Fraud
- Survived or Not Survived

---

# 📖 What is Logistic Regression? (Theory)

Logistic Regression estimates the probability of a data point belonging to a specific class using the **Sigmoid (Logistic) Function**.

The output is always between **0 and 1**, representing a probability.

### Sigmoid Function

:contentReference[oaicite:0]{index=0}

Where:

- **P(y=1)** = Probability of belonging to Class 1
- **e** = Euler's Number
- **z** = Linear Combination of Features

---

## Linear Combination

Before applying the sigmoid function:

:contentReference[oaicite:1]{index=1}

Where:

- **β₀** = Intercept
- **β₁, β₂ ... βₙ** = Coefficients
- **x₁, x₂ ... xₙ** = Features

---

# 🎯 Classification Decision

The predicted probability is converted into a class label using a threshold.

### Example

```text
If Probability ≥ 0.50 → Class 1
If Probability < 0.50 → Class 0
```

---

# 🔢 Types of Logistic Regression

## 1. Binary Logistic Regression

Used when there are only two classes.

### Examples

- Yes / No
- True / False
- Spam / Not Spam
- Survived / Not Survived

---

## 2. Multinomial Logistic Regression

Used when there are more than two classes without any order.

### Examples

- Red, Blue, Green
- Cat, Dog, Bird

---

## 3. Ordinal Logistic Regression

Used when categories have a natural order.

### Examples

- Low, Medium, High
- Poor, Average, Excellent

---

# ✨ Features of Logistic Regression

### Advantages

- Easy to implement and interpret.
- Fast training process.
- Works well for binary classification.
- Provides probability scores.
- Less prone to overfitting for smaller datasets.

### Limitations

- Assumes a linear relationship between features and log-odds.
- Sensitive to outliers.
- Struggles with complex non-linear relationships.
- Performance decreases when classes overlap heavily.

---

# 📋 Assumptions of Logistic Regression

### 1. Binary or Categorical Target Variable
The dependent variable should be categorical.

### 2. Independence of Observations
Observations should not depend on one another.

### 3. No Multicollinearity
Independent variables should not be highly correlated.

### 4. Large Sample Size
Generally performs better with sufficient data.

### 5. Linear Relationship with Log-Odds
Features should have a linear relationship with the log-odds of the target.

---

# ⚙️ Workflow of the Model

## Step 1: Data Collection
Collect the dataset.

## Step 2: Data Preprocessing
- Handle missing values
- Remove duplicates
- Encode categorical variables

## Step 3: Exploratory Data Analysis (EDA)
- Analyze distributions
- Detect outliers
- Study feature relationships

## Step 4: Feature Selection
Select important features.

## Step 5: Feature Scaling (Optional)
Apply scaling if required.

## Step 6: Train-Test Split
Split data into:
- Training Set
- Testing Set

## Step 7: Model Training
Train the Logistic Regression model.

## Step 8: Probability Prediction
Calculate probabilities for each observation.

## Step 9: Classification
Convert probabilities into class labels.

## Step 10: Model Evaluation
Evaluate performance using classification metrics.

---

# 🔄 Workflow Diagram

```text
Data Collection
       ↓
Data Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Feature Selection
       ↓
Feature Scaling
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Probability Prediction
       ↓
Classification
       ↓
Model Evaluation
       ↓
Conclusion
```

---

# 📊 Evaluation Metrics

## Accuracy

Measures the percentage of correct predictions.

:contentReference[oaicite:2]{index=2}

---

## Precision

Measures how many predicted positives are actually positive.

:contentReference[oaicite:3]{index=3}

---

## Recall

Measures how many actual positives are correctly identified.

:contentReference[oaicite:4]{index=4}

---

## F1 Score

Harmonic mean of Precision and Recall.

:contentReference[oaicite:5]{index=5}

---

# 📊 Confusion Matrix

A Confusion Matrix summarizes classification performance.

|               | Predicted Positive | Predicted Negative |
|---------------|-------------------|-------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

---

# 📊 Result

After training the model:

- Probabilities are generated for each observation.
- Class labels are assigned using a threshold.
- Classification metrics evaluate performance.
- Important features influencing predictions can be analyzed.

### Example Output

| Actual | Predicted |
|----------|----------|
| 1 | 1 |
| 0 | 0 |
| 1 | 1 |
| 0 | 1 |

---

# 🎯 Conclusion

Logistic Regression is one of the most widely used classification algorithms. It predicts probabilities using the sigmoid function and converts them into class labels. Due to its simplicity, interpretability, and efficiency, it is often used as a baseline model for classification tasks.

### Key Takeaways

- **Algorithm Type:** Supervised Learning
- **Problem Type:** Classification
- **Output:** Probability + Class Label
- **Function Used:** Sigmoid Function
- **Common Metrics:** Accuracy, Precision, Recall, F1 Score
- **Best Use Case:** Binary and multiclass classification problems