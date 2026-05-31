# Linear Regression (Simple & Multiple Linear Regression)

## 📌 Overview

Linear Regression is a supervised machine learning algorithm used to predict a continuous numerical value based on one or more independent variables. It establishes a linear relationship between input features and the target variable.

Examples:
- Predicting house prices
- Predicting sales revenue
- Predicting employee salaries
- Predicting stock trends

---

# 📖 What is Linear Regression? (Theory)

Linear Regression finds the best-fit line that minimizes the error between actual and predicted values.

The relationship between variables is represented as:

### Simple Linear Regression


::contentReference[oaicite:0]{index=0}


Where:

- **y** = Dependent Variable (Target)
- **x** = Independent Variable (Feature)
- **β₀** = Intercept
- **β₁** = Slope (Coefficient)
- **ε** = Error Term

---

## Types of Linear Regression

### 1. Simple Linear Regression

Uses **one independent variable** to predict the target variable.

#### Example

Predicting Salary based on Years of Experience.

| Experience | Salary |
|------------|---------|
| 1 | 25000 |
| 2 | 30000 |
| 3 | 35000 |

---

### 2. Multiple Linear Regression

Uses **multiple independent variables** to predict the target variable.

#### Formula

:contentReference[oaicite:1]{index=1}

#### Example

Predicting House Price based on:
- Area
- Number of Bedrooms
- Age of House
- Location Score

---

# ✨ Features of Linear Regression

### Advantages

- Easy to understand and implement.
- Fast training process.
- Highly interpretable model.
- Works well when a linear relationship exists.
- Provides insights into feature importance.

### Limitations

- Assumes a linear relationship.
- Sensitive to outliers.
- Can suffer from multicollinearity.
- Performance decreases with complex non-linear patterns.

---

# 📋 Assumptions of Linear Regression

### 1. Linearity
Independent variables should have a linear relationship with the target variable.

### 2. Independence
Observations should be independent of each other.

### 3. Homoscedasticity
Residuals should have constant variance.

### 4. Normality of Residuals
Residuals should follow a normal distribution.

### 5. No Multicollinearity
Independent variables should not be highly correlated.

---

# ⚙️ Workflow of the Model

## Step 1: Data Collection
Collect the dataset.

## Step 2: Data Preprocessing
- Handle missing values
- Remove duplicates
- Fix inconsistencies

## Step 3: Exploratory Data Analysis (EDA)
- Analyze distributions
- Check correlations
- Detect outliers

## Step 4: Feature Selection
Choose relevant independent variables.

## Step 5: Feature Engineering (Optional)
Create or transform features if required.

## Step 6: Train-Test Split
Split data into:
- Training Set
- Testing Set

## Step 7: Model Training
Train the Linear Regression model.

## Step 8: Prediction
Generate predictions on test data.

## Step 9: Model Evaluation
Evaluate using:
- MAE
- MSE
- RMSE
- R² Score

## Step 10: Interpretation
Analyze coefficients and feature impact.

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
Feature Engineering
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Prediction
       ↓
Model Evaluation
       ↓
Interpret Results
       ↓
Conclusion
```

---

# 📊 Evaluation Metrics

## Mean Absolute Error (MAE)

Measures average absolute prediction error.

:contentReference[oaicite:2]{index=2}

---

## Mean Squared Error (MSE)

Measures average squared prediction error.

:contentReference[oaicite:3]{index=3}

---

## Root Mean Squared Error (RMSE)

Square root of MSE.

:contentReference[oaicite:4]{index=4}

---

## R² Score

Measures how much variance is explained by the model.

:contentReference[oaicite:5]{index=5}

---

# 📊 Result

After training the model:

- The best-fit regression line is generated.
- Predictions are made on unseen data.
- Model performance is evaluated using regression metrics.
- Feature coefficients indicate the impact of each variable.

### Example Output

| Actual Value | Predicted Value |
|-------------|----------------|
| 25000 | 25500 |
| 32000 | 31500 |
| 40000 | 39500 |
| 50000 | 51000 |

---

# 🎯 Conclusion

Linear Regression is one of the most fundamental and widely used supervised learning algorithms for predicting continuous values. Simple Linear Regression uses one independent variable, while Multiple Linear Regression uses multiple variables to improve prediction accuracy. When assumptions are satisfied, Linear Regression provides interpretable and reliable results.

### Key Takeaways

- **Algorithm Type:** Supervised Learning
- **Problem Type:** Regression
- **Target Variable:** Continuous Numerical Value
- **Simple Regression:** One Independent Variable
- **Multiple Regression:** Multiple Independent Variables
- **Common Metrics:** MAE, MSE, RMSE, R² Score
- **Best Use Case:** Predicting continuous values with linear relationships