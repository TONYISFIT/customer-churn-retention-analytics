"""Smoke tests for the leakage-safe training pipeline.

Trains on a small sample so the suite stays fast; asserts the pipeline
contract (binary predictions, probability outputs, serialisable metrics).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import TARGET, generate_churn_dataset  # noqa: E402
from preprocessing import build_preprocessor  # noqa: E402
from train import evaluate, get_models  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    df = generate_churn_dataset(n_customers=200, seed=99)
    X = df.drop(columns=["customer_id", TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


def test_preprocessor_output_shape(sample):
    X_train, _, _, _ = sample
    Xt = build_preprocessor().fit_transform(X_train)
    assert Xt.shape[0] == X_train.shape[0]
    assert Xt.shape[1] > X_train.shape[1]  # one-hot expands categoricals


def test_pipeline_predictions_binary(sample):
    X_train, X_test, y_train, _ = sample
    pipe = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=500))]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    assert set(np.unique(preds)) <= {0, 1}
    proba = pipe.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_evaluate_returns_expected_keys(sample):
    X_train, X_test, y_train, y_test = sample
    pipe = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=500))]
    )
    pipe.fit(X_train, y_train)
    metrics = evaluate(y_test, pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1])
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_get_models_includes_baselines():
    names = set(get_models())
    assert {"logistic_regression", "random_forest", "hist_gradient_boosting"} <= names
