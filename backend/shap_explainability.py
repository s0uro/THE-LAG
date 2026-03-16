# pyright: reportGeneralTypeIssues=false
import pandas as pd
from pathlib import Path
from joblib import load

import matplotlib
import numpy as np

# Χρησιμοποιούμε non-GUI backend ώστε να δουλεύει σε server/threads
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap  # pyright: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split


INPUT_FILE = "backend/data_clean.csv"
MODEL_XGB_PATH = "backend/model_xgb.joblib"
SPLIT_INFO_PATH = "backend/split_info.joblib"

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

    # Ίδια προεπεξεργασία features όπως στο training.py / evaluation.py
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

    y: pd.Series = pd.Series(df[TARGET_COL], index=df.index)
    return X, y


def main() -> None:
    _force_utf8_stdio()
    print("Φόρτωση δεδομένων και XGBoost...")
    df = read_input(INPUT_FILE)
    X, y = get_feature_target(df)

    xgb_model = load(MODEL_XGB_PATH)
    split_info = load(SPLIT_INFO_PATH)

    test_size = split_info.get("test_size", 0.2)
    random_state = split_info.get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Για SHAP, χρησιμοποιούμε ένα υποσύνολο για να είναι πιο γρήγορο
    X_train_df = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
    background = X_train_df.sample(min(200, len(X_train_df)), random_state=42)

    print("Δημιουργία SHAP TreeExplainer για XGBoost...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(background)

    # === 1. Συνολική σημασία χαρακτηριστικών (summary plot) ===
    print("Αποθήκευση συνολικού summary plot σε 'shap_summary.png'...")
    plt.figure()
    shap.summary_plot(
        shap_values,
        background,
        show=False,
        plot_type="dot",
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=200)
    plt.close()

    # === 2. Dependence plot (πάντα δημιουργούμε 2ο PNG) ===
    if "Lag" in background.columns:
        feature_for_dependence = "Lag"
        out_name = "shap_dependence_Lag.png"
        print("Αποθήκευση dependence plot για 'Lag' σε 'shap_dependence_Lag.png'...")
    else:
        # Αν δεν υπάρχει Lag, διαλέγουμε το πιο σημαντικό feature
        # βάσει του μέσου |SHAP value|.
        print("Δεν βρέθηκε 'Lag'. Επιλογή πιο σημαντικού feature για dependence plot...")
        shap_array = np.array(shap_values)
        # Για binary classification, shap_values μπορεί να είναι list -> παίρνουμε την πρώτη κλάση
        if isinstance(shap_values, list):
            shap_array = np.array(shap_values[0])
        importances = np.mean(np.abs(shap_array), axis=0)
        top_idx = int(np.argmax(importances))
        feature_for_dependence = background.columns[top_idx]
        out_name = f"shap_dependence_{feature_for_dependence}.png"
        print(f"Αποθήκευση dependence plot για '{feature_for_dependence}' σε '{out_name}'...")

    plt.figure()
    shap.dependence_plot(
        feature_for_dependence,
        shap_values,
        background,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    plt.close()

    print("Ολοκληρώθηκαν τα SHAP γραφήματα (summary + dependence).")


if __name__ == "__main__":
    main()

