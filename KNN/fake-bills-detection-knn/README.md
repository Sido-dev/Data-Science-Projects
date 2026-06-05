# Fake Bills Detection using K-Nearest Neighbors (KNN)

A machine learning classification project that detects whether a bill is genuine or fake using the K-Nearest Neighbors (KNN) algorithm.

## Project Overview
This project:
- Loads and explores the dataset
- Handles missing values
- Scales numerical features using StandardScaler
- Trains a KNN classifier
- Evaluates model performance
- Finds an approximate optimal K value

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Workflow
1. Load dataset (`fake_bills.csv`)
2. Data inspection
3. Missing value handling
4. Feature scaling
5. Train/Test split
6. KNN training
7. Model evaluation
8. Hyperparameter exploration

## Files
```
fake_bills.ipynb
fake_bills.csv
README.md
requirements.txt
```

## Installation
```bash
pip install -r requirements.txt
```

## Run
```bash
jupyter notebook fake_bills.ipynb
```

## Model
Algorithm: KNeighborsClassifier

Example:
```python
knn = KNeighborsClassifier(n_neighbors=6)
```

## Improvements
- Use Pipeline to avoid data leakage
- Tune K using GridSearchCV
- Add ROC-AUC and cross validation
- Save trained model

## License
MIT
