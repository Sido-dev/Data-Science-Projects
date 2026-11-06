# Loan Default Prediction — CatBoost

A reproducible binary-classification experiment with separate training, validation and final test sets. This update replaces the earlier notebook's test-selected threshold and stale saved outputs.

## Setup and run

Tested on Linux x86_64 with Python 3.12.14. Create a separate environment for this project.

```bash
git clone https://github.com/Sido-dev/Data-Science-Projects.git
cd Data-Science-Projects/CatBoost/Loan_Default_Prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. Install JupyterLab separately if you want the notebook interface: `python -m pip install jupyterlab`.

Obtain `Loan_default.csv` from the [dataset page](https://www.kaggle.com/datasets/nikhil1e9/loan-default), or use the existing CSV in this repository at `LightGBM/Loan_Default_Prediction/Loan_default.csv`. Keep the source/license with the data. The CSV is not redistributed in this patch.

```bash
python loan_pipeline.py --csv ../../LightGBM/Loan_Default_Prediction/Loan_default.csv --output artifacts --candidates 4 --folds 3 --threads 2
```

Alternatively open `Loan_Default_Prediction.ipynb`, set `LOAN_CSV` or edit its configuration, then restart the kernel and run all cells. Run either the CLI or notebook for a new experiment; both call the same pipeline. The checked-in notebook contains the completed full-CSV run described below.

## Evaluation contract

- Stratified 60% training / 20% validation / 20% test split, seed 42; exported row indices are disjoint.
- Training-only stratified CV selects CatBoost parameters using ROC-AUC. Four candidates and three folds are the default, not an exhaustive search.
- Validation-only F1 selects a threshold over 0.01–0.99; ties prefer the higher threshold. No error-cost assumptions were supplied, so this is not a business-optimal lending threshold.
- The model and threshold freeze before the final test report. No train+validation refit occurs after threshold selection.
- Report ROC-AUC, average precision, precision, recall, F1 and a confusion matrix together; compare against the majority/prior baseline. Raw accuracy is insufficient for this imbalanced label.
- `LoanID` and target are excluded from features. Duplicate IDs are rejected. Deterministic ratio engineering uses no fitted data statistics. No global imputation/encoding is learned before splitting.

## Verified full-data run

The packaged `artifacts/evaluation.json` is authoritative. The generated run table is inserted during packaging. Previous notebook metrics must not be reused.

Run: 255,347 rows; train 153,207 / validation 51,070 / final test 51,070. Seed 42, four randomized candidates, three training CV folds. These are measured results from the corrected full-CSV run, not estimated improvements.

| Test configuration | Threshold | ROC-AUC | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation-selected threshold | 0.61 | 0.7588 | 0.3318 | 0.2872 | 0.5242 | 0.3711 | 0.7937 |
| Fixed 0.50 threshold | 0.50 | 0.7588 | 0.3318 | 0.2252 | 0.6995 | 0.3408 | 0.6856 |
| Prior/majority baseline | 0.50 | 0.5000 | 0.1161 | 0.0000 | 0.0000 | 0.0000 | 0.8839 |

The selected model uses class balancing, depth 4, 200 iterations, learning rate 0.08 and L2 regularization 3. The threshold was 0.61 on validation, with validation F1 0.3658. The majority baseline has higher accuracy because it predicts no defaults, but its default-class recall/F1 are zero. This update improves evaluation correctness; it does not claim that raw accuracy increased. Old and new metrics are not directly comparable because the split and selection procedure changed.

## Export and inference

`artifacts/` includes `loan_default_model.cbm`, `model_metadata.json`, `evaluation.json`, `validation_thresholds.json`, `cv_results.csv`, `split_indices.npz`, and the notebook's confusion-matrix image. Metadata records the feature order, categorical columns, positive label, threshold and model checksum.

```python
import json
from pathlib import Path
from catboost import CatBoostClassifier
from loan_pipeline import prepare_features

model = CatBoostClassifier()
model.load_model("artifacts/loan_default_model.cbm")
metadata = json.loads(Path("artifacts/model_metadata.json").read_text())
probability = model.predict_proba(prepare_features(new_applications))[:, 1]
predicted_default = probability >= metadata["threshold"]
```

For this snippet, `new_applications` is a DataFrame containing the input columns listed in `loan_pipeline.py`. Verify the saved checksum before loading downloaded model files.

## Limits

The random row split assumes independent borrowers and no temporal deployment gap. Repeated borrowers require group splitting; application dates require time-based evaluation. Exact duplicate IDs are checked, but the data does not establish causal validity. No calibration, fairness, real financial savings or production readiness is claimed. Class weighting can affect probability calibration. Further model selection needs a new untouched holdout.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Tests verify reproducible disjoint partitions, threshold tie behavior, feature transformation, duplicate IDs and an actual CatBoost export/reload.

[Threshold selection reference](https://scikit-learn.org/stable/modules/classification_threshold.html)
