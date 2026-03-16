import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor  # pyright: ignore[reportMissingImports]
from sklearn.neural_network import MLPClassifier, MLPRegressor


INPUT_FILE = "backend/data_clean.csv"

MODEL_XGB_PATH = "backend/model_xgb.joblib"
MODEL_MLP_PATH = "backend/model_mlp.joblib"
MODEL_XGB_REG_PATH = "backend/model_xgb_reg.joblib"
MODEL_MLP_REG_PATH = "backend/model_mlp_reg.joblib"
SCALER_PATH = "backend/scaler.joblib"
SPLIT_INFO_PATH = "backend/split_info.joblib"

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
    """Κοινή λογική encoding για classification και regression."""
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
    """Features + binary target για classification."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Λείπει η στήλη '{TARGET_COL}'.")
    # Αφαιρούμε το Lag από features (data leakage: Lag -> target)
    exclude = [c for c in [TARGET_COL, LAG_COL] if c in df.columns]
    X = _encode_features(df, exclude_cols=exclude)
    y = pd.Series(df[TARGET_COL].values, dtype=df[TARGET_COL].dtype)
    return X, y


def get_feature_lag(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series] | None:
    """Features + συνεχής Temporal Lag για regression."""
    if LAG_COL not in df.columns:
        return None
    lag_series = pd.to_numeric(df[LAG_COL], errors="coerce")
    valid = lag_series.notna()
    df_v = df[valid].copy()
    lag_v = lag_series[valid].reset_index(drop=True)
    exclude = [c for c in [LAG_COL, TARGET_COL] if c in df_v.columns]
    X = _encode_features(df_v, exclude_cols=exclude)
    return X, lag_v


def main() -> None:
    _force_utf8_stdio()
    print(f"Διαβάζω δεδομένα από '{INPUT_FILE}'...")
    df = read_input(INPUT_FILE)

    # ══════════════════════════════════════════════
    # ΜΕΡΟΣ Α — CLASSIFICATION (δύσκολη vs εύκολη λέξη)
    # ══════════════════════════════════════════════
    print("\n" + "="*55)
    print("ΜΕΡΟΣ Α — CLASSIFICATION")
    print("="*55)

    X_clf, y_clf = get_feature_target(df)

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    scaler = StandardScaler()
    Xs_tr_c = scaler.fit_transform(X_tr_c)
    Xs_te_c = scaler.transform(X_te_c)

    xgb_clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    print("Εκπαίδευση XGBoost Classifier...")
    xgb_clf.fit(X_tr_c, y_tr_c)
    yp = xgb_clf.predict(X_te_c)
    print(f"  [XGBoost CLF] Accuracy={accuracy_score(y_te_c,yp):.4f}  F1={f1_score(y_te_c,yp,zero_division=0):.4f}")

    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
        alpha=1e-4, batch_size=64, learning_rate_init=1e-3,  # type: ignore[arg-type]
        max_iter=200, random_state=42, early_stopping=True, n_iter_no_change=10,
    )
    print("Εκπαίδευση MLP Classifier...")
    mlp_clf.fit(Xs_tr_c, y_tr_c)
    yp = mlp_clf.predict(Xs_te_c)
    print(f"  [MLP CLF]     Accuracy={accuracy_score(y_te_c,yp):.4f}  F1={f1_score(y_te_c,yp,zero_division=0):.4f}")

    # ══════════════════════════════════════════════
    # ΜΕΡΟΣ Β — REGRESSION (πρόβλεψη τιμής Temporal Lag)
    # ══════════════════════════════════════════════
    reg_result = get_feature_lag(df)
    has_regression = reg_result is not None

    if has_regression:
        print("\n" + "="*55)
        print("ΜΕΡΟΣ Β — REGRESSION (Temporal Lag σε δευτερόλεπτα)")
        print("="*55)

        X_reg, y_reg = reg_result  # type: ignore[misc]

        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )

        scaler_reg = StandardScaler()
        Xs_tr_r = scaler_reg.fit_transform(X_tr_r)
        Xs_te_r = scaler_reg.transform(X_te_r)

        xgb_reg = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", tree_method="hist", random_state=42,
        )
        print("Εκπαίδευση XGBoost Regressor...")
        xgb_reg.fit(X_tr_r, y_tr_r)
        yp = xgb_reg.predict(X_te_r)
        print(f"  [XGBoost REG] MAE={mean_absolute_error(y_te_r,yp):.4f}s  R²={r2_score(y_te_r,yp):.4f}")

        mlp_reg = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            alpha=1e-4, learning_rate_init=1e-3,
            max_iter=300, random_state=42, early_stopping=True, n_iter_no_change=10,
        )
        print("Εκπαίδευση MLP Regressor...")
        mlp_reg.fit(Xs_tr_r, y_tr_r)
        yp = mlp_reg.predict(Xs_te_r)
        print(f"  [MLP REG]     MAE={mean_absolute_error(y_te_r,yp):.4f}s  R²={r2_score(y_te_r,yp):.4f}")

        dump(xgb_reg, MODEL_XGB_REG_PATH)
        dump(mlp_reg, MODEL_MLP_REG_PATH)
        print(f"\n  Αποθηκεύτηκαν: {MODEL_XGB_REG_PATH}, {MODEL_MLP_REG_PATH}")
    else:
        print("\n[INFO] Δεν βρέθηκε στήλη 'Lag' — παραλείπεται το regression.")

    # Αποθήκευση
    print("\nΑποθήκευση μοντέλων...")
    dump(xgb_clf, MODEL_XGB_PATH)
    dump(mlp_clf, MODEL_MLP_PATH)
    dump(scaler, SCALER_PATH)
    dump({
        "X_columns": X_clf.columns.tolist(),
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
        "has_regression": has_regression,
    }, SPLIT_INFO_PATH)

    print("Ολοκληρώθηκε η εκπαίδευση.")


if __name__ == "__main__":
    main()
