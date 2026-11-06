"""Reproducible CatBoost experiment with train-only tuning and a locked test set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

SEED = 42
NUMERIC = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
           "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"]
CATEGORICAL = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage",
               "HasDependents", "LoanPurpose", "HasCoSigner"]
FEATURES = NUMERIC + ["loan_income_ratio"] + CATEGORICAL


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The same deterministic transformation must be used during inference."""
    missing = set(NUMERIC + CATEGORICAL) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    out = frame[NUMERIC + CATEGORICAL].copy()
    for name in NUMERIC:
        out[name] = pd.to_numeric(out[name], errors="raise").astype(float)
    if np.isinf(out[NUMERIC].to_numpy()).any():
        raise ValueError("Numeric inputs contain infinity")
    # No fitted imputer/scaler is learned outside the training folds.
    out["loan_income_ratio"] = out["LoanAmount"] / out["Income"].where(out["Income"] > 0)
    for name in CATEGORICAL:
        out[name] = out[name].fillna("__MISSING__").astype(str)
    return out[FEATURES]


def split_indices(y: pd.Series, seed: int = SEED):
    """60% train, 20% validation, 20% final test; row IDs never overlap."""
    train_val, test = train_test_split(np.arange(len(y)), test_size=0.2,
                                       stratify=y, random_state=seed)
    train, validation = train_test_split(train_val, test_size=0.25,
        stratify=y.iloc[train_val], random_state=seed)
    assert not (set(train) & set(validation) or set(train) & set(test)
                or set(validation) & set(test))
    return train, validation, test


def choose_threshold(y_validation, probabilities):
    """Maximize validation F1 on a fixed grid; ties prefer the higher threshold."""
    thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    scores = [f1_score(y_validation, probabilities >= t, zero_division=0)
              for t in thresholds]
    best = max(zip(scores, thresholds))
    return float(best[1]), [{"threshold": float(t), "f1": float(s)}
                             for t, s in zip(thresholds, scores)]


def metrics(y, probabilities, threshold):
    predicted = probabilities >= threshold
    return {
        "threshold": float(threshold), "rows": len(y),
        "default_prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, predicted,
            labels=[0, 1], output_dict=True, zero_division=0),
    }


def run(csv_path, output_dir, n_iter=4, folds=3, threads=2, seed=SEED,
        smoke=False):
    csv_path, output = Path(csv_path), Path(output_dir)
    frame = pd.read_csv(csv_path)
    if "LoanID" in frame and frame["LoanID"].duplicated().any():
        raise ValueError("Duplicate LoanID values: resolve borrower overlap before splitting")
    if "Default" not in frame or frame["Default"].isna().any():
        raise ValueError("Default target is missing")
    if set(frame["Default"].unique()) != {0, 1}:
        raise ValueError("Default must contain both binary labels 0 and 1")
    y = frame["Default"].astype(int)
    X = prepare_features(frame)
    train, validation, test = split_indices(y, seed)
    search = RandomizedSearchCV(
        CatBoostClassifier(loss_function="Logloss",
            verbose=False, random_seed=seed, thread_count=threads,
            allow_writing_files=False),
        param_distributions={"iterations": [15] if smoke else [200, 400],
            "depth": [3] if smoke else [4, 6],
            "learning_rate": [0.08] if smoke else [0.03, 0.08],
            "l2_leaf_reg": [3, 7], "auto_class_weights": ["None", "Balanced"]},
        n_iter=n_iter, scoring="roc_auc", n_jobs=1, refit=True,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        random_state=seed, error_score="raise", return_train_score=False)
    # Pass categorical names to fit: avoids CatBoost/sklearn clone incompatibility.
    search.fit(X.iloc[train], y.iloc[train], cat_features=CATEGORICAL)
    model = search.best_estimator_
    validation_probability = model.predict_proba(X.iloc[validation])[:, 1]
    threshold, curve = choose_threshold(y.iloc[validation], validation_probability)
    # Freeze the model, hyperparameters and threshold BEFORE consulting y_test.
    test_probability = model.predict_proba(X.iloc[test])[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(X.iloc[train], y.iloc[train])
    baseline_probability = dummy.predict_proba(X.iloc[test])[:, 1]
    report = {
        "evaluation_status": "synthetic_smoke_only" if smoke else "held_out_test",
        "seed": seed, "dataset_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "selection": {"cv_scoring": "roc_auc", "folds": folds, "candidates": n_iter,
            "best_cv_roc_auc": float(search.best_score_), "best_parameters": search.best_params_,
            "threshold_objective": "validation F1; higher threshold wins ties",
            "validation": metrics(y.iloc[validation], validation_probability, threshold)},
        "test": metrics(y.iloc[test], test_probability, threshold),
        "test_at_0_5": metrics(y.iloc[test], test_probability, 0.5),
        "majority_baseline_test": metrics(y.iloc[test], baseline_probability, 0.5),
        "limitations": ["Random row split assumes independent borrowers and no time ordering.",
            "F1 is a development objective, not a lending cost or fairness assessment.",
            "Do not retune against this test set; use a new holdout for future selection."]}
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output / "loan_default_model.cbm"))
    metadata = {"schema_version": 1, "features": FEATURES,
        "categorical_features": CATEGORICAL, "positive_label": 1,
        "threshold": threshold, "preprocessing": "loan_pipeline.prepare_features",
        "model_sha256": hashlib.sha256((output / "loan_default_model.cbm").read_bytes()).hexdigest(),
        "evaluation_status": report["evaluation_status"]}
    for name, value in [("evaluation.json", report), ("model_metadata.json", metadata),
                        ("validation_thresholds.json", curve)]:
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    pd.DataFrame(search.cv_results_).to_csv(output / "cv_results.csv", index=False)
    # Explicit row manifests allow a reviewer to audit disjoint partitions.
    np.savez_compressed(output / "split_indices.npz", train=train, validation=validation, test=test)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="Loan_default.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    result = run(args.csv, args.output, args.candidates, args.folds, args.threads)
    print(json.dumps(result["test"], indent=2))
