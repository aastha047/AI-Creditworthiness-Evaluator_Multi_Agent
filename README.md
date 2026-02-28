AI Creditworthiness Evaluator – Multi-Agent Agentic AI System

A modular, explainable credit risk evaluation system built using a multi-agent architecture, machine learning, SHAP-based interpretability, and scorecard transformation logic.

This system trains on the UCI Default of Credit Card Clients dataset and allows user-uploaded data for automated credit scoring, decisioning, and explanation.

🏗️ Architecture Overview

The system follows a structured multi-agent design:

- Planner Agent – Orchestrates execution flow
- Data Agent – Preprocesses and validates datasets
- Scoring Agent – Generates default probabilities
- Decision Agent – Converts probabilities into APPROVE / REVIEW / REJECT
- Explain Agent – Produces SHAP-based and LLM-powered explanations
- Feedback Agent – Logs scoring events (audit-ready)

This modular structure improves transparency, maintainability, and scalability.

🧠 Model Details

- Algorithm: RandomForestClassifier
- Trees: 300
- Max Depth: 12
- Accuracy: ~81.7%
- Output: Probability of default (binary classification)

Prediction is generated via:

predict_proba(... )[:, 1]

🚀 Quick Start

1️⃣ Upload Dataset(or any of the same columns)


synthetic_credit_data.csv

upload a CSV/XLSX via Streamlit UI.

2️⃣ Train Model (Optional)

python train_model.py

3️⃣ Launch Streamlit App

streamlit run app.py

Upload dataset → View scores → View decisions → View explanations → Download results

📊 Scoring Formula & Scorecard Mapping

1️⃣ Model Output (Probability)

The model predicts:

prob_default = predict_proba(... )[:, 1]

2️⃣ Simple Legacy Score (0–1000)

score = int((1 - prob_default) * 1000)

Range:

- 0 → Highest Risk
- 1000 → Lowest Risk

3️⃣ Recommended Scorecard (Odds-to-Points)

Implements industry-standard scorecard logic via prob_to_score_card in scoring_agent.py.

Formula:
B = PDO / ln(2)
odds_bad = p / (1 - p)
odds_ref = base_prob / (1 - base_prob)
score = base_score + B * ln(odds_ref / odds_bad)

Default Parameters:

- base_score = 600
- pdo = 20
- base_prob = 0.02

Meaning:

A 2% default probability maps to score 600.

🏛️ Decision Logic

Final decision categories:

- APPROVE
- MANUAL_REVIEW
- REJECT

Thresholds are dynamically computed in app.py using:

- Mean probability
- Standard deviation

Mapping logic implemented in decision_agent.py.

📦 Tech Stack

- Python
- scikit-learn
- SHAP
- Streamlit
- SQLite
- pandas / numpy
- Pickle serialization

🔎 Explainability

- SHAP-based feature attribution
- Top contributing factors displayed
- Optional LLM narrative explanation
- Full audit trail support

Designed for regulatory transparency and responsible AI.
