

# AI Creditworthiness Evaluator (Agentic AI)

Train on the UCI dataset and allow user uploads for credit scoring and explanations.

## Quick Start
1. Place the dataset `default_of_credit_card_clients.xls` (or another CSV/XLSX) in the repository root.
2. Train the model (optional):

```powershell
python train_model.py
```

3. Launch the Streamlit app:

```powershell
streamlit run app.py
```

4. Upload a CSV or Excel dataset in the app and view scores, explanations and decisions.

**Scoring Formula & Configuration**

- **Model Output (probability):** The model predicts the probability of default (`prob_default`) using `predict_proba(... )[:, 1]` (see `scoring_agent.py`).

- **Simple Legacy Score (0–1000):** Earlier code mapped the probability to a score using:

   ```python
   score = int((1 - prob_default) * 1000)
   ```

   This yields a score from ~0 (worst) to 1000 (best).

- **Recommended Scorecard (Odds-to-Points):** The repository also implements a standard scorecard mapping that transforms probability into log-odds and then into points using a "Points-to-Double-Odds" (PDO) parameter. This is implemented as `prob_to_score_card` in `scoring_agent.py`.

   Formula used in code:

   - B = PDO / ln(2)
   - odds_bad = p / (1 - p)
   - odds_ref = base_prob / (1 - base_prob)
   - score = base_score + B * ln(odds_ref / odds_bad)

   Defaults in code: `base_score=600`, `pdo=20`, `base_prob=0.02`. With these defaults, a 2% default probability maps to score 600.

- **Decision Logic:** Decisions (APPROVE / MANUAL_REVIEW / REJECT) are made from the raw predicted probability `p` using thresholds computed in `app.py` from the predictions' mean and standard deviation. The decision mapping is implemented in `decision_agent.py`.

## Customize Score Mapping

- Change parameters where `prob_to_score_card` is called in `app.py` if you want a different anchor or sensitivity:

   - `base_score` — reference score at `base_prob` (default 600).
   - `pdo` — points to double odds (commonly 20 or 40).
   - `base_prob` — reference default probability (e.g., 0.02).

- Example (Python):

   ```python
   from scoring_agent import prob_to_score_card
   score = prob_to_score_card(0.12, base_score=650, pdo=20, base_prob=0.05)
   print(score)
   ```

## Next Steps (optional)

- I can add interactive Streamlit controls so users can change `base_score`, `pdo`, and `base_prob` from the UI.
- I can implement a fully auditable scorecard pipeline (WOE binning + logistic regression) that produces per-feature points and a transparent scoring table — recommended for production.

If you want either of those, tell me which and I will implement it.
