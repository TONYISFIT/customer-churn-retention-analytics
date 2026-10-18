"""Permutation importance for the best churn pipeline.

Computed on the held-out test set with the raw (pre-preprocessing) features,
so importances map back to the original business features. This is an honest,
model-agnostic alternative to SHAP; adding SHAP values is listed as future work.

Outputs:
  - reports/feature_importance.csv
  - reports/feature_importance.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import TARGET  # noqa: E402

RANDOM_STATE = 42


def main() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "churn_customers.csv")
    feature_cols = [c for c in df.columns if c not in ("customer_id", TARGET)]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        df[TARGET],
        test_size=0.20,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )

    pipe = joblib.load(ROOT / "models" / "churn_pipeline.joblib")
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="f1"
    )
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    importance.to_csv(reports / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        importance["feature"][::-1],
        importance["importance_mean"][::-1],
        xerr=importance["importance_std"][::-1],
        color="#1f77b4",
    )
    ax.set_xlabel("Mean decrease in F1 (10 permutations)")
    ax.set_title("Permutation importance — churn model (test set)")
    fig.tight_layout()
    fig.savefig(reports / "feature_importance.png", dpi=150)
    plt.close(fig)

    print(importance.to_string(index=False))
    return importance


if __name__ == "__main__":
    main()
