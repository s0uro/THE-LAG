import pandas as pd
from pathlib import Path
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier  # pyright: ignore[reportMissingImports]
from sklearn.neural_network import MLPClassifier


INPUT_FILE = "backend/data_clean.csv"  # έξοδος από script1_preprocessing

MODEL_XGB_PATH = "backend/model_xgb.joblib"
MODEL_MLP_PATH = "backend/model_mlp.joblib"
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

    # Αντιγράφουμε για να μην αλλάζουμε το αρχικό DataFrame
    df = df.copy()

    # Μετατροπή όλων των feature στηλών σε αριθμητικές / κωδικοποιημένες κατηγορίες
    for col in df.columns:
        if col == TARGET_COL:
            continue

        if df[col].dtype == "object":
            # Προσπάθεια για numeric
            numeric = pd.to_numeric(df[col], errors="coerce")
            non_na_ratio = numeric.notna().mean()

            if non_na_ratio > 0.5:
                # Αν η πλειοψηφία είναι νούμερα, κράτα την numeric εκδοχή
                df[col] = numeric
            else:
                # Αλλιώς, κάνε το κατηγορική και βάλε codes
                df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=[TARGET_COL])

    # Ασφάλεια: καμία στήλη του X να μην μείνει ως object
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes

    # Αφαίρεση NaN για να μπορεί να εκπαιδευτεί το MLP
    X = X.fillna(0)

    y = df[TARGET_COL]
    return X, y


def main() -> None:
    _force_utf8_stdio()
    print(f"Διαβάζω προεπεξεργασμένα δεδομένα από '{INPUT_FILE}'...")
    df = read_input(INPUT_FILE)

    X, y = get_feature_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # === XGBoost Classifier ===
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )

    print("Εκπαίδευση XGBoost...")
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb)
    print(f"[XGBoost] Accuracy: {acc_xgb:.4f}, F1: {f1_xgb:.4f}")

    # === MLP Classifier ===
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
    )

    print("Εκπαίδευση MLP...")
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp)
    print(f"[MLP]     Accuracy: {acc_mlp:.4f}, F1: {f1_mlp:.4f}")

    # Αποθήκευση μοντέλων και πληροφορίας split (για scripts 4 & 5)
    print("Αποθήκευση εκπαιδευμένων μοντέλων...")
    dump(xgb_model, MODEL_XGB_PATH)
    dump(mlp_model, MODEL_MLP_PATH)

    print("Αποθήκευση πληροφοριών split (για evaluation/SHAP)...")
    split_info = {
        "X_columns": X.columns.tolist(),
        "test_size": 0.2,
        "random_state": 42,
        "stratify": True,
    }
    dump(split_info, SPLIT_INFO_PATH)

    print("Ολοκληρώθηκε η εκπαίδευση και η αποθήκευση των μοντέλων.")


if __name__ == "__main__":
    main()

