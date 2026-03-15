import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


INPUT_FILE = "backend/data_clean.csv"
MODEL_XGB_PATH = "backend/model_xgb.joblib"
MODEL_MLP_PATH = "backend/model_mlp.joblib"
SPLIT_INFO_PATH = "backend/split_info.joblib"
METRICS_JSON_PATH = "metrics.json"  # θα γραφτεί στον κεντρικό φάκελο του project

TARGET_COL = "target"

def _force_utf8_stdio() -> None:
    import sys

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def read_input(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".xlsx":
        return pd.read_excel(path)
    elif ext == ".csv":
        # low_memory=False για να αποφύγουμε DtypeWarning με mixed types
        return pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Μη υποστηριζόμενο format αρχείου: {ext}")


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise KeyError(f"Λείπει η στήλη target ('{TARGET_COL}') από τα δεδομένα.")

    df = df.copy()

    # Ίδια προεπεξεργασία features όπως στο training.py
    for col in df.columns:
        if col == TARGET_COL:
            continue

        if df[col].dtype == "object":
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_na_ratio = float(pd.Series(numeric).notna().mean())
            if non_na_ratio > 0.5:
                df[col] = numeric
            else:
                df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=[TARGET_COL])

    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype) == "string":
            X[col] = X[col].astype("category").cat.codes
        elif hasattr(X[col], "cat"):
            X[col] = X[col].cat.codes
        if X[col].dtype not in ["int64","float64","int32","float32","bool","int8","int16"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(0)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes

    y: pd.Series = pd.Series(df[TARGET_COL])
    return X, y


def main() -> None:
    _force_utf8_stdio()
    print("Φόρτωση δεδομένων και μοντέλων...")
    df = read_input(INPUT_FILE)
    X, y = get_feature_target(df)

    xgb_model = load(MODEL_XGB_PATH)
    mlp_model = load(MODEL_MLP_PATH)
    split_info = load(SPLIT_INFO_PATH)

    # Επαναχρησιμοποιούμε το ίδιο random_state/test_size ώστε να έχουμε ίδιο split
    test_size = split_info.get("test_size", 0.2)
    random_state = split_info.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # === Αξιολόγηση XGBoost ===
    print("=== Αξιολόγηση XGBoost ===")
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    # zero_division=0 για να μη βγάζει UndefinedMetricWarning όταν δεν υπάρχουν θετικές προβλέψεις
    f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)  # type: ignore[arg-type]
    report_xgb = classification_report(
        y_test,
        y_pred_xgb,
        output_dict=True,
        zero_division=0,  # type: ignore[arg-type]
    )
    cm_xgb = confusion_matrix(y_test, y_pred_xgb).tolist()

    print(f"Accuracy: {acc_xgb:.4f}")
    print(f"F1-Score: {f1_xgb:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred_xgb, digits=4, zero_division=0))  # type: ignore[arg-type]
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_xgb))

    # === Αξιολόγηση MLP ===
    print("\n=== Αξιολόγηση MLP ===")
    y_pred_mlp = mlp_model.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    # zero_division=0 για να μη βγάζει UndefinedMetricWarning όταν δεν υπάρχουν θετικές προβλέψεις
    f1_mlp = f1_score(y_test, y_pred_mlp, zero_division=0)  # type: ignore[arg-type]
    report_mlp = classification_report(
        y_test,
        y_pred_mlp,
        output_dict=True,
        zero_division=0,  # type: ignore[arg-type]
    )
    cm_mlp = confusion_matrix(y_test, y_pred_mlp).tolist()

    print(f"Accuracy: {acc_mlp:.4f}")
    print(f"F1-Score: {f1_mlp:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred_mlp, digits=4, zero_division=0))  # type: ignore[arg-type]
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_mlp))

    # === Αποθήκευση metrics σε JSON για το frontend ===
    metrics = {
        "xgboost": {
            "accuracy": acc_xgb,
            "f1": f1_xgb,
            "report": report_xgb,
            "confusion_matrix": cm_xgb,
        },
        "mlp": {
            "accuracy": acc_mlp,
            "f1": f1_mlp,
            "report": report_mlp,
            "confusion_matrix": cm_mlp,
        },
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

