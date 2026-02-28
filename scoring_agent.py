import pickle
import pandas as pd
import numpy as np

class ScoringAgent:
    def __init__(self, model_path="model.pkl"):
        with open(model_path, "rb") as f:
            obj = pickle.load(f)

        self.model = obj["model"]
        self.features = obj["features"]

    def predict(self, X):
        Xp = X.copy()
        for col in Xp.select_dtypes(include=["object"]).columns:
            Xp[col] = Xp[col].astype("category").cat.codes

        Xp = Xp.fillna(0)
        Xp = Xp.reindex(columns=self.features, fill_value=0)

        preds = self.model.predict_proba(Xp)[:, 1]
        return preds.tolist()


# --------------------------
#  FIXED Scorecard Formula
#  (NO DASHBOARD PARAMETERS)
# --------------------------
def prob_to_score(prob):
    """
    Convert p(default) → credit score (300–900)
    reputable & research‑friendly mapping
    """
    p = float(prob)
    p = max(1e-12, min(1 - 1e-12, p))

    base_score = 650        # reference point
    base_prob = 0.08        # probability at base score (8% = industry avg)
    pdo = 50                # every +50 score = risk halves

    B = pdo / np.log(2)
    odds_bad = p / (1 - p)
    odds_ref = base_prob / (1 - base_prob)

    score = base_score + B * np.log(odds_ref / odds_bad)
    return int(round(max(300, min(900, score))))
