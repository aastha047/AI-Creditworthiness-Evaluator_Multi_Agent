import json
import requests
from config import GROK_API_KEY, GROK_API_URL, GROQ_MODEL

class PlannerAgent:
    def __init__(self):
        pass

    def call_grok(self, goal, context):
        # Best-effort local plan if GROK not configured
        if not GROK_API_KEY or not GROK_API_URL:
            return (
                "1. Preprocess and validate the input data.\n"
                "2. Score applicants with the trained model.\n"
                "3. Explain top SHAP features for the selected applicant.\n"
                "4. Produce a decision and log feedback."
            )
        prompt = (
            f"Goal: {goal}\nContext: {json.dumps(context, indent=2)}\n\n"
            "Provide a short, numbered plan (2-6 steps) to evaluate creditworthiness for the given sample."
        )
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a concise planner. Output numbered steps only."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300,
            "temperature": 0.2
        }
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        try:
            resp = requests.post(GROK_API_URL, headers=headers, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return f"Groq error: {resp.status_code} - {resp.text}"
        except Exception as e:
            return f"Groq call failed: {e}"
