import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from loan_pipeline import CATEGORICAL, NUMERIC, choose_threshold, prepare_features, run, split_indices


def synthetic_frame(rows=160):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({name: rng.uniform(1, 100, rows) for name in NUMERIC})
    for name in CATEGORICAL:
        frame[name] = np.where(np.arange(rows) % 2, "yes", "no")
    frame["Default"] = (np.arange(rows) % 4 == 0).astype(int)
    frame["LoanID"] = [f"id-{i}" for i in range(rows)]
    return frame


def test_disjoint_reproducible_stratified_splits():
    y = synthetic_frame()["Default"]
    partitions = split_indices(y)
    assert [len(p) for p in partitions] == [96, 32, 32]
    assert len(set(np.concatenate(partitions))) == len(y)
    assert all(np.array_equal(a, b) for a, b in zip(partitions, split_indices(y)))
    assert all(y.iloc[p].mean() == 0.25 for p in partitions)


def test_threshold_uses_validation_and_matches_grid_maximum():
    y, scores = [0, 0, 1, 1], np.array([0.05, 0.15, 0.3, 0.8])
    threshold, curve = choose_threshold(y, scores)
    assert 0.15 < threshold <= 0.3
    assert max(row["f1"] for row in curve) == 1.0
    assert threshold == 0.3  # deterministic tie rule


def test_feature_schema_and_invalid_income():
    frame = synthetic_frame(8)
    frame.loc[0, "Income"] = 0
    frame.loc[0, CATEGORICAL[0]] = None
    features = prepare_features(frame)
    assert "Default" not in features and "LoanID" not in features
    assert np.isnan(features.loc[0, "loan_income_ratio"])
    assert features.loc[0, CATEGORICAL[0]] == "__MISSING__"


def test_real_catboost_export_round_trip(tmp_path):
    frame = synthetic_frame()
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    report = run(path, tmp_path / "out", n_iter=1, folds=2, threads=1, smoke=True)
    assert report["evaluation_status"] == "synthetic_smoke_only"
    assert sum(map(sum, report["test"]["confusion_matrix"])) == 32
    loaded = CatBoostClassifier()
    loaded.load_model(str(tmp_path / "out/loan_default_model.cbm"))
    assert loaded.predict_proba(prepare_features(frame.head())).shape == (5, 2)


def test_duplicate_borrower_rejected(tmp_path):
    import pytest
    frame = synthetic_frame()
    frame.loc[1, "LoanID"] = frame.loc[0, "LoanID"]
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate LoanID"):
        run(path, tmp_path / "out", smoke=True)
