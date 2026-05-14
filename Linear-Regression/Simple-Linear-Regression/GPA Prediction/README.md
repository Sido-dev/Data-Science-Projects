# Simple Linear Regression – GPA Prediction Using SAT Scores

## Overview

This project demonstrates the implementation of **Simple Linear Regression** to predict student GPA based on SAT scores. The workflow includes data preprocessing, outlier handling using winsorization, exploratory data analysis, model training, prediction, and performance evaluation.

The project highlights how linear regression can be used to identify and model relationships between variables for predictive analysis.

---

## Objective

* Predict GPA using SAT scores
* Understand the relationship between SAT and GPA
* Improve model performance by handling lower extreme outliers
* Evaluate regression performance using standard metrics

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Workflow

1. Importing required libraries
2. Loading and understanding the dataset
3. Exploratory Data Analysis (EDA)
4. Detecting outliers using the IQR method
5. Applying lower-side winsorization on GPA values
6. Visualizing the relationship using scatter plots
7. Splitting the dataset into training and testing sets
8. Building the Simple Linear Regression model
9. Predicting GPA values
10. Evaluating model performance using:

* RMSE
* R² Score

---

## Outlier Handling

Lower extreme GPA values were capped using winsorization to reduce the influence of outliers and improve model stability.

* Model accuracy before capping: **41%**
* Model accuracy after capping: **48.66%**

This demonstrates that handling outliers improved the overall performance of the regression model.

---

## Results

The model achieved:

* Improved regression stability
* Better prediction performance after outlier treatment
* A moderate positive linear relationship between SAT scores and GPA

---

## Conclusion

This project successfully demonstrates the complete workflow of building a Simple Linear Regression model for predictive analytics. By applying preprocessing techniques such as winsorization, the model performance improved significantly, making the predictions more reliable and stable.

---

