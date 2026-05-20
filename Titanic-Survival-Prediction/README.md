# Titanic Survival Prediction

Binary classification to predict which passengers survived the Titanic disaster using the Kaggle Titanic dataset.

---

## Kaggle Score

| Model | Score |
|---|---|
| Logistic Regression | **0.76315** |

---

## Project Structure

```
Titanic-Survival-Prediction/
├── titanic_survival_prediction.ipynb
├── train.csv
├── test.csv
├── gender_submission.csv
└── README.md
```

---

## Workflow

1. **Load Data** — train, test, and submission template CSVs
2. **Exploratory Data Analysis** — inspect column types, missing values, target distribution
3. **Preprocess** — drop high-cardinality columns (`Name`, `Ticket`, `Cabin`), one-hot encode `Sex` and `Embarked`
4. **Impute Missing Values** — fill `Age` with median, `Fare` with median
5. **Feature Engineering** — invert Age (`100 - Age`) so higher value = younger passenger
6. **Model Training** — Logistic Regression via scikit-learn
7. **Predict & Submit** — generate predictions on test set, export `submission.csv`

---

## Results

- **Training Accuracy:** ~80%
- **Kaggle Public Score:** 0.76315
- Prediction distribution closely matched training label distribution (~38% survived)

---

## Tools & Libraries

| Library | Purpose |
|---|---|
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Model training and evaluation |

---

## Dataset

From the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

| File | Description |
|---|---|
| `train.csv` | 891 passengers with survival labels |
| `test.csv` | 418 passengers without labels |
| `gender_submission.csv` | Submission template |

---

## How to Run

1. Clone the repo
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn`
3. Place `train.csv`, `test.csv`, `gender_submission.csv` in the same folder as the notebook
4. Run all cells in `titanic_survival_prediction.ipynb`
5. Submit `submission.csv` to Kaggle
