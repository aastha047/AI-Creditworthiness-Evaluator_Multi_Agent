import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Import agents
from data_agent import DataAgent
from scoring_agent import ScoringAgent
from explain_agent import ExplainAgent
from decision_agent import DecisionAgent
from feedback_agent import FeedbackAgent
from planner_agent import PlannerAgent

from config import MODEL_PATH
from scipy.stats import rankdata

st.set_page_config(page_title="AI Credit Evaluator", layout="wide")
st.title("AI Creditworthiness Evaluator")

# ---------------------------
# File Upload Section
# ---------------------------
uploaded_file = st.file_uploader(
    "📤 Upload dataset (CSV / Excel)", type=["csv", "xls", "xlsx"]
)
retrain = st.checkbox("🔄 Retrain model on uploaded data (optional)", value=False)

# ---------------------------
# Ensure Model Exists
# ---------------------------
def ensure_model_exists(X=None, y=None):
    if not Path(MODEL_PATH).exists():
        if X is None or y is None:
            return None
        model_obj = RandomForestClassifier(n_estimators=100, random_state=42)
        model_obj.fit(X, y)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model_obj, "features": X.columns.tolist()}, f)
        return model_obj
    else:
        with open(MODEL_PATH, "rb") as f:
            saved = pickle.load(f)
        return saved["model"]

# ---------------------------
# Top-5 Explanation Factors
# ---------------------------
def generate_top5_reasons(sample_row: pd.DataFrame, X_all: pd.DataFrame):
    r = sample_row.iloc[0]
    reasons = []

    pay_cols = [c for c in X_all.columns if c.upper().startswith("PAY_") and not c.upper().startswith("PAY_AMT")]
    if pay_cols:
        pref = "PAY_1" if "PAY_1" in pay_cols else pay_cols[0]
        val = r.get(pref, None)
        if pd.notnull(val) and float(val) > 0:
            reasons.append({"feature": f"Recent payment ({pref})", "direction": "Increases Risk", "explanation": "Delayed or missed recent payments increase risk."})
        else:
            reasons.append({"feature": f"Recent payment ({pref})", "direction": "Reduces Risk", "explanation": "No recent payment delays observed."})

    bill_cols = [c for c in X_all.columns if c.upper().startswith("BILL_AMT")]
    if bill_cols:
        median_bill = X_all[bill_cols].mean(axis=1).median()
        avg_bill_row = r[bill_cols].mean()
        if avg_bill_row > median_bill:
            reasons.append({"feature": "High bill amount", "direction": "Increases Risk", "explanation": "Applicant has larger debt exposure than peers."})
        else:
            reasons.append({"feature": "Bill amounts", "direction": "Reduces Risk", "explanation": "Lower outstanding bills compared to peers."})

    if "AGE" in X_all.columns:
        median_age = X_all["AGE"].median()
        age_val = r.get("AGE", np.nan)
        if pd.notnull(age_val) and age_val > median_age:
            reasons.append({"feature": "Age", "direction": "Reduces Risk", "explanation": "Older applicants show more stable financial behaviour."})
        else:
            reasons.append({"feature": "Age", "direction": "Increases Risk", "explanation": "Younger applicants may have limited credit history."})

    if "LIMIT_BAL" in X_all.columns:
        median_limit = X_all["LIMIT_BAL"].median()
        limit_val = r.get("LIMIT_BAL", np.nan)
        if limit_val < median_limit:
            reasons.append({"feature": "Credit Limit", "direction": "Increases Risk", "explanation": "Lower credit limit indicates tighter financial capacity."})
        else:
            reasons.append({"feature": "Credit Limit", "direction": "Reduces Risk", "explanation": "Higher credit limit offers financial cushion."})

    payamt_cols = [c for c in X_all.columns if c.upper().startswith("PAY_AMT")]
    if payamt_cols:
        median_payamt = X_all[payamt_cols].mean(axis=1).median()
        avg_pay_row = r[payamt_cols].mean()
        if avg_pay_row > median_payamt:
            reasons.append({"feature": "Past Payments", "direction": "Reduces Risk", "explanation": "Strong repayment record."})
        else:
            reasons.append({"feature": "Past Payments", "direction": "Increases Risk", "explanation": "Lower repayment behaviour compared to peers."})

    return reasons[:5]

def highlight_rows(row):
    if row["Direction"] == "Increases Risk":
        return ["background-color: #ffe6e6"] * len(row)
    if row["Direction"] == "Reduces Risk":
        return ["background-color: #e6ffe6"] * len(row)
    return [""] * len(row)

# ---------------------------
# MAIN APP LOGIC
# ---------------------------
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Preprocess
    data_agent = DataAgent()
    X, y = data_agent.preprocess(df)
    st.success("Data cleaned successfully ✔")

    # Planner Agent
    try:
        planner = PlannerAgent()
        planner.call_grok("Evaluate creditworthiness", {"rows": len(X)})
    except:
        pass

    # Train or Load Model
    if retrain and y is not None:
        model_obj = RandomForestClassifier(n_estimators=400, max_depth=14, random_state=42, n_jobs=-1)
        model_obj.fit(X, y)
        calibrated_model = CalibratedClassifierCV(model_obj, cv='prefit', method='sigmoid')
        calibrated_model.fit(X, y)
        model = calibrated_model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "features": X.columns.tolist()}, f)
        st.success("Model retrained and calibrated successfully.")
    else:
        model = ensure_model_exists(X=X, y=y)
        if model is None:
            st.error("Cannot load model. Enable retraining or upload labels.")
            st.stop()

    # Scoring
    scoring_agent = ScoringAgent()
    preds = scoring_agent.predict(X)

    # ---------------------------
    # New Realistic Score Conversion (percentiles)
    # ---------------------------
    percentiles = rankdata(1 - np.array(preds), method="average") / len(preds)
    scores = np.round(300 + percentiles * 600)  # 300-900 score range

    # Decisions based on realistic thresholds
    mean_p = np.mean(preds)
    std_p = np.std(preds)
    approve_threshold = max(0.2, mean_p - std_p)
    manual_review_threshold = min(0.6, mean_p + std_p)

    decision_agent = DecisionAgent(
        approve_threshold=approve_threshold,
        manual_review_threshold=manual_review_threshold,
    )

    # Row-level display
    row_index = st.number_input(
        "🔍 Choose row for explanation",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
    )

    sample_row = X.iloc[[int(row_index)]]
    prob_default = float(preds[int(row_index)])
    score = int(scores[int(row_index)])
    decision = decision_agent.make_decision(prob_default)

    # Explanations
    top_reasons_clean = generate_top5_reasons(sample_row, X)
    expl_df = pd.DataFrame(top_reasons_clean)
    expl_df.columns = ["Factor", "Direction", "Explanation"]

    st.subheader(f"Top 5 Factors Influencing Applicant #{row_index}")
    st.dataframe(expl_df.style.apply(highlight_rows, axis=1))

    explain_agent = ExplainAgent(MODEL_PATH)
    score_obj = {"score": score, "prob_default": prob_default, "decision": decision}
    llm_text = explain_agent.grok_explanation(score_obj, top_reasons_clean)

    st.subheader("Human-Style Explanation (Grok LLM)")
    st.write(llm_text)

    # Table
    results = []
    for i, p in enumerate(preds):
        results.append({"Record": i + 1, "Credit Score": int(scores[i]), "Prob Default": round(p, 4), "Decision": decision_agent.make_decision(p)})

    result_df = pd.DataFrame(results)
    st.subheader("Predictions & Decisions for All Rows")
    st.dataframe(result_df)

    st.download_button(
        "⬇ Download Results CSV",
        data=result_df.to_csv(index=False),
        file_name="credit_risk_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("Upload a dataset (CSV/XLSX) to begin.")
