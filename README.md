# Customer Churn Prediction

Predict whether a telecom customer will churn, using leakage-safe
scikit-learn pipelines, a real model comparison, and an interactive
Streamlit demo.

> **All data in this repository is synthetic** — generated with a fixed
> random seed (`data/generate_churn_data.py`) for learning and demo
> purposes. It is not real customer data.

## Problem

Customer churn is expensive: acquiring a new customer costs far more than
retaining one. This project builds a binary classifier that flags
customers likely to churn so a (hypothetical) retention team could act.

## Approach

1. **Synthetic data generation** (`src/dataset.py`) — 2,000 telecom
   customers with tenure, charges, contract type, payment method,
   internet service, and support-call history. Churn labels come from a
   logistic data-generating process (month-to-month contracts, fiber
   internet, and frequent support calls raise churn risk), so the data
   carries real, learnable signal. Churn rate: **33.2%**.
2. **Leakage-safe preprocessing** (`src/preprocessing.py`) — a
   `ColumnTransformer` (standard-scaled numerics, one-hot categoricals)
   embedded *inside* each `Pipeline`, so encoding/scaling statistics are
   learned from training folds only.
3. **Model comparison** (`src/train.py`) — Logistic Regression,
   Random Forest, HistGradientBoosting, and XGBoost, evaluated with
   stratified 5-fold CV plus a held-out 20% test split. Metrics are
   written to `reports/metrics.json` by the training script itself.
4. **Interpretability** (`src/importance.py`) — permutation importance
   on the test set (model-agnostic; SHAP is listed as future work).
5. **Demo app** (`app.py`) — Streamlit UI for single-customer scoring
   and batch CSV scoring.

## Results

Metrics below are the **actual outputs of `python src/train.py`**
(seed 42, 80/20 stratified split, 5-fold CV). Nothing here is hand-written.

| Model | CV F1 (mean ± std) | Test Acc | Test Prec | Test Rec | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4956 ± 0.0260 | 0.7275 | 0.6463 | 0.3985 | 0.4930 | 0.7675 |
| Random Forest | 0.4075 ± 0.0306 | 0.6975 | 0.5625 | 0.4060 | 0.4716 | 0.7233 |
| HistGradientBoosting | 0.4350 ± 0.0317 | 0.7050 | 0.5701 | 0.4586 | 0.5083 | 0.7257 |
| XGBoost | 0.4165 ± 0.0450 | 0.7075 | 0.5889 | 0.3985 | 0.4753 | 0.7402 |

Best model by test F1: **HistGradientBoosting** (saved to
`models/churn_pipeline.joblib`).

Top features by permutation importance (mean F1 decrease, test set):

| Feature | Importance |
|---|---|
| tenure_months | 0.092 |
| total_charges | 0.063 |
| support_calls | 0.062 |
| internet_service | 0.046 |
| contract_type | 0.040 |
| monthly_charges | 0.036 |
| payment_method | 0.026 |

Full per-class reports: `reports/classification_report.txt`.
Feature ranking CSV: `reports/feature_importance.csv`.

## How to run

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python data/generate_churn_data.py

# 2. Train + compare models (writes reports/ and models/)
python src/train.py

# 3. Permutation importance (writes reports/feature_importance.csv/.png)
python src/importance.py

# 4. Launch the demo app (train first so the pipeline exists)
streamlit run app.py

# 5. Run the test suite
pytest -q
```

CI runs `pytest` on Python 3.10 and 3.11 (`.github/workflows/ci.yml`).

## Limitations

- The dataset is **synthetic**; results demonstrate methodology, not real
  customer behaviour.
- Class imbalance and noisy labels keep F1 modest (~0.5) — realistic for
  a first baseline, and an honest starting point for iteration
  (threshold tuning, calibration, richer features).
- The trained pipeline (`models/churn_pipeline.joblib`) is a local
  build artifact, not versioned in git; retrain with `src/train.py`.
- The Streamlit app is a **demo**, not production retention advice.

## Ethics note

Churn models can be misused to penalise vulnerable customers. Any real
deployment should be reviewed for fairness (e.g. disparate impact across
customer segments), used only to offer help/retention incentives, and
governed by the applicable data-protection rules.

## Future work

- SHAP values for per-prediction explanations
- Probability calibration + threshold tuning for retention-campaign ROI
- Experiment tracking (MLflow)

## License

MIT — see [LICENSE](LICENSE).
