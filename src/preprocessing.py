"""Preprocessing: leakage-safe column transformer for the churn dataset.

The transformer is always embedded inside an sklearn Pipeline (see train.py)
so that scaling / encoding statistics are learned from training folds only.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # package-style import (tests, library use)
    from .dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES
except ImportError:  # script-style import (src/ on sys.path)
    from dataset import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer for numeric + categorical churn features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
