"""Streamlit demo: customer churn prediction.

Single-customer form + batch CSV scoring using the saved pipeline.
Run:  streamlit run app.py   (after training: python src/train.py)
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
INTERNET = ["DSL", "Fiber optic", "No"]


@st.cache_resource
def load_pipeline():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.title("Customer Churn Prediction")
    st.caption(
        "Demo on **synthetic** telecom data. Scores are illustrative, "
        "not production retention advice."
    )

    pipe = load_pipeline()
    if pipe is None:
        st.error(
            "No trained model found. Train first with `python src/train.py`, "
            "then restart the app."
        )
        st.stop()

    tab1, tab2 = st.tabs(["Single customer", "Batch scoring"])

    with tab1:
        st.subheader("Predict churn for one customer")
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly charges ($)", 0.0, 500.0, 70.0)
            total = st.number_input("Total charges ($)", 0.0, 20000.0, 840.0)
            calls = st.number_input("Support calls (last 6 months)", 0, 20, 1)
        with col2:
            contract = st.selectbox("Contract type", CONTRACTS)
            payment = st.selectbox("Payment method", PAYMENTS)
            internet = st.selectbox("Internet service", INTERNET)

        if st.button("Predict churn"):
            row = pd.DataFrame(
                [
                    {
                        "tenure_months": tenure,
                        "monthly_charges": monthly,
                        "total_charges": total,
                        "support_calls": calls,
                        "contract_type": contract,
                        "payment_method": payment,
                        "internet_service": internet,
                    }
                ]
            )
            proba = float(pipe.predict_proba(row)[0, 1])
            pred = int(pipe.predict(row)[0])
            st.metric("Churn probability", f"{proba:.1%}")
            st.write("**Prediction:**", "Likely to churn" if pred else "Likely to stay")
            st.progress(proba)

    with tab2:
        st.subheader("Score a CSV of customers")
        st.caption(
            "Upload a CSV with columns: tenure_months, monthly_charges, "
            "total_charges, support_calls, contract_type, payment_method, "
            "internet_service."
        )
        uploaded = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded is not None:
            batch = pd.read_csv(uploaded)
            proba = pipe.predict_proba(batch)[:, 1]
            batch["churn_probability"] = proba.round(4)
            batch["predicted_churn"] = (proba >= 0.5).astype(int)
            st.dataframe(batch.head(20))
            st.download_button(
                "Download scored CSV",
                batch.to_csv(index=False),
                file_name="churn_scored.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
