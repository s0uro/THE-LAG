# pyright: reportGeneralTypeIssues=false
import json
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


INPUT_FILE = "backend/data_clean.csv"
MODEL_XGB_PATH = "backend/model_xgb.joblib"
MODEL_MLP_PATH = "backend/model_mlp.joblib"
MODEL_XGB_REG_PATH = "backend/model_xgb_reg.joblib"
MODEL_MLP_REG_PATH = "backend/model_mlp_reg.joblib"
SCALER_PATH = "backend/scaler.joblib"
SPLIT_INFO_PATH = "backend/split_info.joblib"
METRICS_JSON_PATH = "metrics.json"

TARGET_COL = "target"
LAG_COL = "Lag"


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
        return pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Μη υποστηριζόμενο format αρχείου: {ext}")


def _encode_features(df: pd.DataFrame, exclude_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == "object" or str(df[col].dtype) == "string":
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_na_ratio = float(pd.Series(numeric).notna().mean())
            if non_na_ratio > 0.5:
                df[col] = numeric
            else:
                df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=exclude_cols, errors="ignore")

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
    return X


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise KeyError(f"Λείπει η στήλη '{TARGET_COL}'.")
    exclude = [c for c in [TARGET_COL, LAG_COL] if c in df.columns]
    X = _encode_features(df, exclude_cols=exclude)
    y = pd.Series(df[TARGET_COL])
    return X, y


def get_feature_lag(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:
    if LAG_COL not in df.columns:
        return None
    lag_series: pd.Series = pd.to_numeric(df[LAG_COL], errors="coerce")  # type: ignore[assignment]
    valid = lag_series.notna()
    df_v = df[valid].copy()
    lag_v = lag_series[valid].reset_index(drop=True)  # type: ignore[call-arg]
    exclude = [c for c in [LAG_COL, TARGET_COL] if c in df_v.columns]
    X = _encode_features(df_v, exclude_cols=exclude)  # type: ignore[arg-type]
    return X, lag_v


def main() -> None:
    _force_utf8_stdio()
    print("Φόρτωση δεδομένων και μοντέλων...")
    df = read_input(INPUT_FILE)

    split_info = load(SPLIT_INFO_PATH)
    test_size = split_info.get("test_size", 0.2)
    random_state = split_info.get("random_state", 42)
    has_regression = split_info.get("has_regression", False)

    metrics: dict = {}

    # ══════════════════════════════════════════════
    # ΜΕΡΟΣ Α — CLASSIFICATION
    # ══════════════════════════════════════════════
    print("\n" + "="*55)
    print("ΑΞΙΟΛΟΓΗΣΗ CLASSIFICATION")
    print("="*55)

    X_clf, y_clf = get_feature_target(df)
    xgb_clf = load(MODEL_XGB_PATH)
    mlp_clf = load(MODEL_MLP_PATH)

    # Scaler για MLP
    scaler = None
    try:
        scaler = load(SCALER_PATH)
        has_scaler = True
    except Exception:
        has_scaler = False

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )

    Xs_te_c = scaler.transform(X_te_c) if has_scaler and scaler is not None else X_te_c  # type: ignore[union-attr]

    # XGBoost
    yp_xgb = xgb_clf.predict(X_te_c)
    acc_xgb = accuracy_score(y_te_c, yp_xgb)
    f1_xgb = f1_score(y_te_c, yp_xgb, zero_division=0)  # type: ignore[arg-type]
    cm_xgb = confusion_matrix(y_te_c, yp_xgb).tolist()
    report_xgb = classification_report(y_te_c, yp_xgb, output_dict=True, zero_division=0)  # type: ignore[arg-type]

    print(f"\n[XGBoost] Accuracy={acc_xgb:.4f}  F1={f1_xgb:.4f}")
    print(classification_report(y_te_c, yp_xgb, digits=4, zero_division=0))  # type: ignore[arg-type]
    print("Confusion Matrix:\n", confusion_matrix(y_te_c, yp_xgb))

    # MLP
    yp_mlp = mlp_clf.predict(Xs_te_c)
    acc_mlp = accuracy_score(y_te_c, yp_mlp)
    f1_mlp = f1_score(y_te_c, yp_mlp, zero_division=0)  # type: ignore[arg-type]
    cm_mlp = confusion_matrix(y_te_c, yp_mlp).tolist()
    report_mlp = classification_report(y_te_c, yp_mlp, output_dict=True, zero_division=0)  # type: ignore[arg-type]

    print(f"\n[MLP]     Accuracy={acc_mlp:.4f}  F1={f1_mlp:.4f}")
    print(classification_report(y_te_c, yp_mlp, digits=4, zero_division=0))  # type: ignore[arg-type]
    print("Confusion Matrix:\n", confusion_matrix(y_te_c, yp_mlp))

    metrics["classification"] = {
        "xgboost": {"accuracy": acc_xgb, "f1": f1_xgb, "report": report_xgb, "confusion_matrix": cm_xgb},
        "mlp":     {"accuracy": acc_mlp, "f1": f1_mlp, "report": report_mlp, "confusion_matrix": cm_mlp},
    }
    # Παλιά δομή για συμβατότητα με το frontend
    metrics["xgboost"] = metrics["classification"]["xgboost"]
    metrics["mlp"] = metrics["classification"]["mlp"]

    # ══════════════════════════════════════════════
    # ΜΕΡΟΣ Β — REGRESSION
    # ══════════════════════════════════════════════
    reg_result = get_feature_lag(df)

    if has_regression and reg_result is not None:
        print("\n" + "="*55)
        print("ΑΞΙΟΛΟΓΗΣΗ REGRESSION (Temporal Lag)")
        print("="*55)

        xgb_reg_path = Path(MODEL_XGB_REG_PATH)
        mlp_reg_path = Path(MODEL_MLP_REG_PATH)

        if xgb_reg_path.exists() and mlp_reg_path.exists():
            X_reg, y_reg = reg_result
            xgb_reg = load(MODEL_XGB_REG_PATH)
            mlp_reg = load(MODEL_MLP_REG_PATH)

            X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
                X_reg, y_reg, test_size=test_size, random_state=random_state
            )

            Xs_te_r = scaler.transform(X_te_r) if has_scaler and scaler is not None else X_te_r  # type: ignore[union-attr]

            yp_xgb_r = xgb_reg.predict(X_te_r)
            mae_xgb = mean_absolute_error(y_te_r, yp_xgb_r)
            r2_xgb = r2_score(y_te_r, yp_xgb_r)
            print(f"\n[XGBoost REG] MAE={mae_xgb:.4f}s  R²={r2_xgb:.4f}")

            yp_mlp_r = mlp_reg.predict(Xs_te_r)
            mae_mlp = mean_absolute_error(y_te_r, yp_mlp_r)
            r2_mlp = r2_score(y_te_r, yp_mlp_r)
            print(f"[MLP REG]     MAE={mae_mlp:.4f}s  R²={r2_mlp:.4f}")

            metrics["regression"] = {
                "xgboost": {"mae": mae_xgb, "r2": r2_xgb},
                "mlp":     {"mae": mae_mlp, "r2": r2_mlp},
            }
        else:
            print("[INFO] Τα regression μοντέλα δεν βρέθηκαν. Τρέξε πρώτα το training.py.")
    else:
        print("\n[INFO] Δεν υπάρχει 'Lag' στα δεδομένα ή regression μοντέλα — παραλείπεται.")

    # Αποθήκευση metrics
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nΑποθηκεύτηκαν metrics στο '{METRICS_JSON_PATH}'")


if __name__ == "__main__":
    main()
