# ❤️ Heart Disease Classification

A Machine Learning project that predicts the likelihood of heart disease using patient health and lifestyle information. The project utilizes a **Random Forest Classifier** and includes data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, and evaluation.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help healthcare professionals identify high-risk individuals and take preventive measures.

This project applies Machine Learning techniques to classify whether a person is likely to have heart disease based on various health-related attributes.

---

## 🎯 Objective

Build a classification model capable of predicting heart disease using patient information such as:

- Age
- Gender
- Blood Pressure
- Cholesterol Level
- Smoking Habits
- Alcohol Consumption
- Family History
- Physical Activity
- Other health indicators

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## 📂 Project Workflow

### 1. Import Libraries
- Load required Python libraries for data analysis, visualization, and machine learning.

### 2. Load Dataset
- Read the dataset into a Pandas DataFrame.

### 3. Data Exploration
- Dataset overview
- Data types
- Statistical summary

### 4. Data Preprocessing
- Missing value analysis
- Duplicate record check
- Feature encoding

### 5. Exploratory Data Analysis (EDA)
- Target variable analysis
- Feature distribution analysis
- Correlation heatmap
- Outlier detection

### 6. Feature Selection
- Identify important features
- Prepare feature matrix and target variable

### 7. Train-Test Split
- Split dataset into training and testing sets

### 8. Model Building
- Random Forest Classifier

### 9. Model Evaluation
- Accuracy Score
- Classification Report
- Training Accuracy
- Testing Accuracy

### 10. Hyperparameter Tuning
- GridSearchCV

### 11. Final Model Performance
- Train final optimized model
- Evaluate performance on unseen data

---

## 📊 Feature Engineering

### Binary Encoding

The following columns were converted from Yes/No values into numerical values:

| Feature | Encoding |
|----------|----------|
| Smoking | Yes → 1, No → 0 |
| Alcohol Consumption | Yes → 1, No → 0 |
| Family History | Yes → 1, No → 0 |

### Ordinal Encoding

| Physical Activity | Encoding |
|------------------|-----------|
| Low | 0 |
| Medium | 1 |
| High | 2 |

### One-Hot Encoding

- Gender

---

## 🤖 Model Used

### Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

#### Advantages
- High accuracy
- Handles complex relationships
- Reduces overfitting
- Robust to noise
- Provides feature importance

---

## 📈 Model Performance

| Metric | Score |
|----------|----------|
| Training Accuracy | 90.38% |
| Testing Accuracy | 83.50% |

The model demonstrates strong predictive performance and good generalization on unseen data.

---

## 📋 Evaluation Metrics

- Accuracy Score
- Classification Report
- Training Accuracy
- Testing Accuracy

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/Sido-dev/heart-disease-classification.git
```

### Navigate to Project

```bash
cd heart-disease-classification
```


### Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```bash
prediction.ipynb
```

---

## 📁 Project Structure

```text
Heart-Disease-Classification/
│
├── prediction.ipynb
├── disease_prediction.csv
├── README.md
```

---

## 🎯 Results

The optimized Random Forest model achieved:

- Training Accuracy: 90.38%
- Testing Accuracy: 83.50%

These results indicate that the model can effectively identify individuals at risk of heart disease while maintaining good generalization performance.

---

## 📝 Conclusion

A Random Forest Classifier was developed to predict heart disease using patient health and lifestyle data. After preprocessing, feature engineering, exploratory data analysis, and hyperparameter tuning, the final model achieved **83.50% testing accuracy**.

The results demonstrate the effectiveness of machine learning in healthcare applications and highlight its potential for supporting early heart disease risk assessment and preventive decision-making.

---

## 👨‍💻 Author

**Sudhanshu Narayane**

If you found this project useful, consider giving it a ⭐ on GitHub.