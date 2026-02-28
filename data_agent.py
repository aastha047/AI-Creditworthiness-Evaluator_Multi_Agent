import pandas as pd
from pathlib import Path

class DataAgent:
    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path is not None else None

    def load_raw(self):
        if self.data_path is None:
            raise ValueError("No data_path provided to DataAgent.")
        p = self.data_path
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found at {p}")
        if p.suffix.lower() in (".xls", ".xlsx"):
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def preprocess(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("preprocess expects a pandas DataFrame")
        df = df.copy().dropna(how="all")
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        target_candidates = [c for c in df.columns if "default" in c.lower() or c.lower() in ("label","target")]
        if target_candidates:
            target_col = target_candidates[0]
            df = df.rename(columns={target_col:"label"})
            y = df["label"].copy()
            X = df.drop(columns=["label"])
        else:
            X = df
            y = None
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category").cat.codes
        X = X.fillna(0)
        return X, y
