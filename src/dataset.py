"""Synthetic telecom customer-churn dataset generator.

All data produced by this module is SYNTHETIC and generated from a fixed
random seed for reproducibility. It is intended for learning/demo purposes
only and must not be treated as real customer data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service"]
TARGET = "churn"


def generate_churn_dataset(n_customers: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic telecom customer dataset with a churn label.

    Churn probability is a logistic function of the features, so the
    generated data carries real, learnable signal:
      - higher churn for month-to-month contracts, fiber-optic internet,
        electronic-check payments, many support calls, high monthly charges
      - lower churn for long tenure and two-year contracts
    """
    rng = np.random.default_rng(seed)

    customer_id = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]

    # Tenure in months: skewed towards shorter tenures (typical for churn data)
    tenure_months = np.minimum(
        rng.exponential(scale=18, size=n_customers).astype(int), 72
    )

    monthly_charges = np.round(rng.normal(loc=65, scale=22, size=n_customers), 2)
    monthly_charges = np.clip(monthly_charges, 18.0, 150.0)

    total_charges = np.round(
        monthly_charges * tenure_months + rng.normal(0, 40, n_customers), 2
    )
    total_charges = np.clip(total_charges, 0.0, None)

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_customers,
        p=[0.55, 0.25, 0.20],
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_customers,
        p=[0.35, 0.20, 0.20, 0.25],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_customers,
        p=[0.35, 0.45, 0.20],
    )
    support_calls = rng.poisson(lam=1.8, size=n_customers)
    support_calls = np.clip(support_calls, 0, 10)

    # Logistic model for churn probability (the "true" data-generating process)
    logit = (
        -2.2
        - 0.045 * tenure_months
        + 0.012 * monthly_charges
        + 0.28 * support_calls
        + 0.9 * (contract_type == "Month-to-month")
        - 0.5 * (contract_type == "Two year")
        + 0.6 * (internet_service == "Fiber optic")
        + 0.4 * (payment_method == "Electronic check")
    )
    proba = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n_customers) < proba).astype(int)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "support_calls": support_calls,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    df = generate_churn_dataset()
    print(df.shape)
    print(df["churn"].value_counts(normalize=True).round(3))
    print(df.head())
