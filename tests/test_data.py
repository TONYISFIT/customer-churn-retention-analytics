"""Sanity checks for the synthetic churn dataset generator."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    generate_churn_dataset,
)


def test_shape_and_columns():
    df = generate_churn_dataset(n_customers=2000, seed=42)
    expected = {"customer_id"} | set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | {TARGET}
    assert df.shape == (2000, 9)
    assert set(df.columns) == expected


def test_no_nulls():
    df = generate_churn_dataset(n_customers=500, seed=7)
    assert df.isnull().sum().sum() == 0


def test_label_binary_and_balanced():
    df = generate_churn_dataset(n_customers=2000, seed=42)
    assert set(df[TARGET].unique()) <= {0, 1}
    rate = df[TARGET].mean()
    assert 0.15 < rate < 0.45, f"unexpected churn rate {rate}"


def test_customer_id_unique():
    df = generate_churn_dataset(n_customers=500, seed=7)
    assert df["customer_id"].is_unique


def test_reproducible_with_seed():
    a = generate_churn_dataset(n_customers=100, seed=123)
    b = generate_churn_dataset(n_customers=100, seed=123)
    assert a.equals(b)
