from pathlib import Path
UCI_DATA_PATH = r"dataset\default of credit card clients.xlsx"
GROK_API_KEY = ""  # your actual key
GROK_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant" 
MODEL_PATH = Path("model.pkl")
DB_PATH = Path("agent_feedback.db")
