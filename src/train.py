"""Train and compare churn classifiers with leakage-safe pipelines.

Every model is wrapped in an sklearn Pipeline whose first step is the
preprocessor, so encoding/scaling statistics are fit on training data only
(no leakage from validation/test folds).

Outputs (all computed from the actual run, nothing hand-written):
  - models/churn_pipeline.joblib      : best pipeline by test F1
  - reports/metrics.json              : CV + holdout metrics per model
  - reports/classification_report.txt : sklearn classification reports
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import TARGET  # noqa: E402
from preprocessing import build_preprocessor  # noqa: E402

RANDOM_STATE = 42
DATA_PATH = ROOT / "data" / "churn_customers.csv"

# XGBoost is optional: include it only if the package is installed.
try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False
    print("xgboost not installed - training without XGBClassifier.")


def get_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if _HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return models


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main() -> dict:
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("customer_id", TARGET)]
    X, y = df[feature_cols], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train={X_train.shape} test={X_test.shape} churn_rate={y.mean():.3f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict = {}
    fitted: dict = {}

    for name, model in get_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        cv_scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1
        )
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, pipe.predict(X_test), proba)
        results[name] = {
            "cv_f1_mean": round(float(cv_scores["test_score"].mean()), 4),
            "cv_f1_std": round(float(cv_scores["test_score"].std()), 4),
            "test": test_metrics,
            "classification_report": classification_report(y_test, pipe.predict(X_test)),
        }
        fitted[name] = pipe
        print(f"{name:24s} cv_f1={results[name]['cv_f1_mean']:.4f} test={test_metrics}")

    best_name = max(results, key=lambda n: results[n]["test"]["f1"])
    print(f"best model by test F1: {best_name}")

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    joblib.dump(fitted[best_name], models_dir / "churn_pipeline.joblib")
    with open(models_dir / "best_model.txt", "w") as f:
        f.write(f"{best_name}\n")

    serialisable = {
        name: {k: v for k, v in info.items() if k != "classification_report"}
        for name, info in results.items()
    }
    serialisable["best_model"] = best_name
    serialisable["random_state"] = RANDOM_STATE
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    with open(reports_dir / "classification_report.txt", "w") as f:
        for name, info in results.items():
            f.write(f"=== {name} ===\n{info['classification_report']}\n")

    return serialisable


if __name__ == "__main__":
    main()
