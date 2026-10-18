#!/usr/bin/env python3
"""Generate the synthetic churn dataset CSV.

SYNTHETIC DATA - generated with a fixed seed for reproducibility.
Run from the repository root:  python data/generate_churn_data.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import generate_churn_dataset

if __name__ == "__main__":
    df = generate_churn_dataset(n_customers=2000, seed=42)
    out = ROOT / "data" / "churn_customers.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} with shape {df.shape}")
    print("churn rate:", round(df["churn"].mean(), 3))
