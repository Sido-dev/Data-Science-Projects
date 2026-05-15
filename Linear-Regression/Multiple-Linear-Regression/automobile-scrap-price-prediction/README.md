# 🚗 Automobile Scrap Price Prediction

> Predicting used/scrap car prices using **Multiple Linear Regression** on real-world automobile specifications.

---

## 📌 Overview

This project builds a **Multiple Linear Regression (MLR)** model to predict the resale/scrap price of automobiles based on their physical and mechanical specifications. The pipeline covers the full data science workflow — from raw data and EDA to feature engineering, encoding, model training, and evaluation.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | `scrap_price.csv` |
| **Records** | 205 vehicles |
| **Features** | 25 (raw) → 23 (after engineering) |
| **Target** | `price` (USD) |
| **Price Range** | $5,118 — $45,400 |

### Key Features Used

`enginesize` · `horsepower` · `curbweight` · `carwidth` · `wheelbase` · `highwaympg` · `cylindernumber` · `Brand_tier` · `carbody` · `drivewheels` · `fuelsystem` · `enginetype` · `aspiration` · and more

---

## 🗂️ Project Workflow

```
Raw Data → EDA → Feature Engineering → Encoding → Train/Test Split → MLR → Evaluation
```

### Steps

1. **Load & Inspect** — Shape, dtypes, null counts, unique values
2. **EDA** — Correlation heatmap to identify low-signal features
3. **Drop Low-Correlation Features** — Removed `symboling`, `carheight`, `stroke`, `compressionratio`, `peakrpm`, `carlength`, `boreratio`, `citympg`
4. **Feature Engineering** — Extracted brand from `name`; fixed typos; mapped brands to a 3-tier system:
   - `2` → Luxury *(Porsche, Jaguar, BMW, Alfa Romeo, Audi)*
   - `1` → Mid *(Volvo, Saab, Volkswagen, Buick)*
   - `0` → Economy *(Toyota, Honda, Nissan, Mazda, Dodge)*
5. **Encoding**
   - Ordinal: `cylindernumber`, `doornumbers`
   - Label Encoding: `fueltypes`, `aspiration`, `enginelocation`, `enginetype`
   - One-Hot Encoding: `carbody`, `drivewheels`, `fuelsystem`
6. **Train / Test Split** — 80% / 20%, `random_state=42`
7. **Model Training** — Scikit-learn `LinearRegression`
8. **Evaluation** — R², MAE, RMSE, MAPE

---

## 📈 Model Performance

```
╔══════════════════════════════════════════════╗
║       FINAL MODEL PERFORMANCE SUMMARY        ║
╠══════════════════════════════════════════════╣
║  R²  Accuracy          :   85.91 %           ║
║  MAE                   :  $   2,405          ║
║  RMSE                  :  $   3,335          ║
║  MAPE                  :    20.32 %          ║
║  Train / Test Split    :   80 % / 20 %       ║
║  Random State          :   42                ║
║  Features used         :   23                ║
╚══════════════════════════════════════════════╝
```

---

## 💡 Observations

- **R² ≈ 0.86** → The model explains ~86% of price variance — strong for a plain OLS regression with no regularisation.
- **MAE of $2,405** is reasonable given the wide price range of $5,118–$45,400.
- **MAPE of 20.32%** is driven by larger percentage errors on cheaper vehicles where even small dollar gaps translate to high relative misses.
- **Brand Tier** was a high-impact engineered feature — luxury brands (Porsche, Jaguar, BMW) command significant price premiums well captured by the tier mapping.
- **Residuals are slightly heteroscedastic** (larger errors at high prices); a log-transform on `price` could stabilise variance.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading & manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | EDA & visualisation |
| `scikit-learn` | Model training & evaluation |

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/your-username/automobile-scrap-price-prediction.git
cd automobile-scrap-price-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 3. Run the notebook
jupyter notebook scrap_price_prediction.ipynb
```

---

## 📁 Repository Structure

```
├── scrap_price.csv                  # Raw dataset
├── scrap_price_prediction.ipynb     # Main notebook
└── README.md                        # Project documentation
```

---
