# train_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

DATA_PATH = r"dataset\default of credit card clients.xlsx"
MODEL_PATH = "model.pkl"


def load_data(path=DATA_PATH):
    df = pd.read_excel(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def preprocess(df):
    df = df.copy()

    # Remove ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Detect target column
    target_candidates = [c for c in df.columns if 'default payment next month' in c.lower()]
    if len(target_candidates) == 0:
        raise ValueError("No target column found in dataset.")

    # Normalize target column name
    df = df.rename(columns={target_candidates[0]: 'label'})

    # Encode object/category columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    # Handle missing values
    df = df.fillna(0)

    return df


# -----------------------------
# 3. Feature Engineering
# -----------------------------
def feature_engineer(df):
    out = df.copy()

    bill_cols = [c for c in out.columns if c.upper().startswith("BILL_AMT")]
    pay_cols = [c for c in out.columns if c.upper().startswith("PAY_AMT")]

    # Add basic aggregations
    if bill_cols and pay_cols:
        out["avg_bill"] = out[bill_cols].mean(axis=1)
        out["avg_pay"] = out[pay_cols].mean(axis=1)
        out["pay_to_bill_ratio"] = out["avg_pay"] / (out["avg_bill"] + 1e-9)

    # Utilization feature
    if "LIMIT_BAL" in out.columns:
        out["utilization"] = out["avg_bill"] / (out["LIMIT_BAL"] + 1e-9)

    # Delay count from payment status
    pay_status = [c for c in out.columns if c.upper().startswith("PAY_") and not c.upper().startswith("PAY_AMT")]
    if pay_status:
        delays = [(out[c] > 0).astype(int) for c in pay_status]
        out["delay_count"] = pd.concat(delays, axis=1).sum(axis=1)

    return out


# -----------------------------
# 4. Model Training
# -----------------------------
def train(save_model=True):
    print("\nLoading dataset...")
    df = load_data()

    print("Preprocessing data...")
    df = preprocess(df)

    print("Applying feature engineering...")
    df = feature_engineer(df)

    # Split X/Y
    X = df.drop(columns=['label'])
    y = df['label'].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    preds_prob = model.predict_proba(X_test)[:, 1]
    preds = (preds_prob > 0.5).astype(int)

    # Evaluation
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds_prob)
    report = classification_report(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\nAccuracy: {acc*100:.2f} %")
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)
    print(f"\nAUC Score: {auc:.4f}")

    # Save model + features
    if save_model:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'model': model,
                'features': X.columns.tolist()
            }, f)
        print(f"\nModel saved to {MODEL_PATH}")

    return model


# -----------------------------
# 5. Entry Point
# -----------------------------
if __name__ == "__main__":
    train()
