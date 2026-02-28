# agents/explain_agent.py
import pickle
import shap
import pandas as pd
import json
import requests
from config import GROK_API_KEY, GROK_API_URL, GROQ_MODEL, MODEL_PATH

class ExplainAgent:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.features = None
        try:
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                self.model = obj.get("model")
                self.features = obj.get("features")
            else:
                self.model = obj
        except Exception as e:
            print("Model load failed in ExplainAgent:", e)

    def shap_top(self, X_row, model=None, n_top=10):
        """
        Return shap values (internal). We keep this but the app will convert to the
        human-friendly Top-5 reasons. This method returns a list of dicts:
        [{"feature": feat_name, "shap": value}, ...]
        """
        model_obj = model if model is not None else self.model
        if model_obj is None:
            raise ValueError("No model provided to shap_top")

        # Align columns if features known
        if self.features:
            X_row = X_row.reindex(columns=self.features, fill_value=0)

        X_row = X_row.copy()
        for c in X_row.columns:
            if X_row[c].dtype == object:
                X_row[c] = pd.to_numeric(X_row[c], errors="coerce").fillna(0)

        explainer = None
        shap_values = None
        try:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_row)
        except Exception:
            # fallback
            try:
                explainer = shap.KernelExplainer(model_obj.predict_proba, X_row)
                shap_values = explainer.shap_values(X_row)
            except Exception as e:
                raise RuntimeError("SHAP explanation failed: " + str(e))

        # Normalize output into list of (feature, value)
        if isinstance(shap_values, list):
            # multiclass case: pick the predicted class
            try:
                probs = model_obj.predict_proba(X_row)[0]
                cls_idx = int(probs.argmax())
            except Exception:
                cls_idx = -1
            vals = shap_values[cls_idx][0] if cls_idx >= 0 else shap_values[-1][0]
        else:
            if shap_values.ndim == 3:
                vals = shap_values[-1][0]
            else:
                vals = shap_values[0]

        vals = vals.flatten()
        items = sorted(zip(list(X_row.columns), vals), key=lambda x: abs(x[1]), reverse=True)[:n_top]
        return [{"feature": f, "shap": float(v)} for f, v in items]

    def grok_explanation(self, score_obj, top_reasons):
        """
        Use the exact prompt specified by the user to generate the human-style explanation.
        top_reasons should be a list/dict suitable for JSON dumping - we pass them into the prompt.
        """
        # Build the exact prompt the user provided (verbatim structure)
        prompt = (
            f"You are an AI assistant that explains credit risk predictions in simple, concise terms.\n"
            f"Credit Score: {score_obj.get('score')}\n"
            f"Probability of Default: {score_obj.get('prob_default', 0):.3f}\n"
            f"Top contributing factors:\n{json.dumps(top_reasons, indent=2)}\n"
            f"Model Decision: {score_obj.get('decision', 'N/A')}\n"
            "Explain in 4-5 short sentences why this applicant got this score using plain, simple language. "
            "Do NOT guess or change the decision — repeat the exact decision given above in the final sentence clearly."
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a concise assistant that explains model outputs simply."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
        try:
            r = requests.post(GROK_API_URL, json=payload, headers=headers, timeout=15)
            if r.status_code == 200:
                # return the assistant content text
                return r.json()["choices"][0]["message"]["content"]
            return "LLM error: " + r.text
        except Exception as e:
            return "LLM call failed: " + str(e)