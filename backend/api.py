from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
import sqlite3
import subprocess
import sys
import time

import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from joblib import load
from pydantic import BaseModel
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent  # THE-LAG root
BACKEND_DIR = BASE_DIR / "backend"

MODEL_XGB_PATH = BACKEND_DIR / "model_xgb.joblib"
MODEL_MLP_PATH = BACKEND_DIR / "model_mlp.joblib"
SPLIT_INFO_PATH = BACKEND_DIR / "split_info.joblib"
METRICS_JSON_PATH = BASE_DIR / "metrics.json"
SHAP_SUMMARY_PATH = BASE_DIR / "shap_summary.png"
DB_PATH = BASE_DIR / "thelag.db"


class PredictRequest(BaseModel):
    """
    Περιμένουμε ένα dict με features:
    {
      "len_scaled": 0.3,
      "freq_scaled": 0.8,
      ...
    }
    Τα ονόματα πρέπει να ταιριάζουν με τα columns που
    χρησιμοποιήθηκαν στο training (βλ. split_info['X_columns']).
    """

    features: Dict[str, Any]


class PredictResponse(BaseModel):
    xgb_prediction: float
    xgb_proba: Optional[float] = None
    mlp_prediction: float
    mlp_proba: Optional[float] = None


class RegisterRequest(BaseModel):
    firstName: str
    lastName: str
    username: str
    email: str
    password: str
    agreement: bool


class LoginRequest(BaseModel):
    username: str
    password: str


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _get_db_connection():
    """Connect with long timeout and WAL-friendly settings to reduce 'database is locked'."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.execute("PRAGMA busy_timeout = 30000")  # wait up to 30 seconds for lock
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _init_db():
    """Create SQLite DB and users table (compatible with DB Browser for SQLite)."""
    conn = _get_db_connection()
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                agreement INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


app = FastAPI(title="THE-LAG ML API")

# Επιτρέπουμε κλήσεις από browser (π.χ. index.html) με CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    _init_db()


# Lazy φορτώση μοντέλων ώστε το API να μπορεί να ξεκινήσει
# ακόμη και αν δεν έχουν εκπαιδευτεί ακόμα.
XGB_MODEL: Optional[Any] = None
MLP_MODEL: Optional[Any] = None
EXPECTED_COLUMNS: Optional[list[str]] = None


def _load_models_and_info(force_reload: bool = False):
    """
    Φορτώνει/επιστρέφει τα μοντέλα και τις στήλες features.
    Αν δεν υπάρχουν τα αρχεία, σηκώνει HTTPException 500 (κατάλληλο για API).
    """
    global XGB_MODEL, MLP_MODEL, EXPECTED_COLUMNS

    if (
        not force_reload
        and XGB_MODEL is not None
        and MLP_MODEL is not None
        and EXPECTED_COLUMNS is not None
    ):
        return XGB_MODEL, MLP_MODEL, EXPECTED_COLUMNS

    if not MODEL_XGB_PATH.exists() or not MODEL_MLP_PATH.exists() or not SPLIT_INFO_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Δεν βρέθηκαν τα εκπαιδευμένα μοντέλα ή το split_info. "
                "Τρέξε πρώτα την εκπαίδευση (ή το upload-and-run)."
            ),
        )

    xgb_model = load(MODEL_XGB_PATH)
    mlp_model = load(MODEL_MLP_PATH)
    split_info = load(SPLIT_INFO_PATH)

    expected_columns = split_info.get("X_columns")
    if expected_columns is None:
        raise HTTPException(status_code=500, detail="Το split_info δεν περιέχει 'X_columns'.")

    XGB_MODEL = xgb_model
    MLP_MODEL = mlp_model
    EXPECTED_COLUMNS = expected_columns

    return XGB_MODEL, MLP_MODEL, EXPECTED_COLUMNS


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Δίνεις features (dict) και παίρνεις προβλέψεις από XGBoost & MLP.
    """
    if not req.features:
        raise HTTPException(status_code=400, detail="Λείπουν features στο request.")

    xgb_model, mlp_model, expected_columns = _load_models_and_info()

    # Φτιάχνουμε DataFrame με ένα μόνο row
    df = pd.DataFrame([req.features])

    # Προσθέτουμε όποιες στήλες λείπουν, με default 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Κρατάμε μόνο τις στήλες που περιμένουν τα μοντέλα, με σωστή σειρά
    df = df[expected_columns]

    # Ασφάλεια: μετατροπή αντικειμένων σε κατηγορικά codes όπως στο training
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    df = df.fillna(0)

    # Προβλέψεις
    xgb_pred = float(xgb_model.predict(df)[0])
    mlp_pred = float(mlp_model.predict(df)[0])

    xgb_proba: Optional[float] = None
    mlp_proba: Optional[float] = None

    if hasattr(xgb_model, "predict_proba"):
        proba = xgb_model.predict_proba(df)[0]
        # Αν είναι binary classification, κρατάμε την πιθανότητα για κλάση 1
        if len(proba) == 2:
            xgb_proba = float(proba[1])

    if hasattr(mlp_model, "predict_proba"):
        proba = mlp_model.predict_proba(df)[0]
        if len(proba) == 2:
            mlp_proba = float(proba[1])

    return PredictResponse(
        xgb_prediction=xgb_pred,
        xgb_proba=xgb_proba,
        mlp_prediction=mlp_pred,
        mlp_proba=mlp_proba,
    )


@app.get("/metrics")
def get_metrics():
    """
    Επιστρέφει τα στατιστικά από το metrics.json (evaluation.py).
    """
    if not METRICS_JSON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Δεν βρέθηκε το metrics.json. Τρέξε πρώτα 'python backend/evaluation.py'.",
        )

    import json

    with METRICS_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/shap-summary")
def shap_summary():
    """
    Επιστρέφει το shap_summary.png ως εικόνα.
    """
    if not SHAP_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Δεν βρέθηκε το shap_summary.png. Τρέξε πρώτα 'python backend/shap_explainability.py'.",
        )

    return FileResponse(str(SHAP_SUMMARY_PATH), media_type="image/png", filename="shap_summary.png")


@app.post("/upload-and-run")
def upload_and_run(file: UploadFile = File(...)):
    """
    Δέχεται ένα αρχείο (xlsx ή csv), το αποθηκεύει ως data.xlsx/data.csv
    στο root του project και τρέχει διαδοχικά:
    - backend/preprocessing.py
    - backend/training.py
    - backend/evaluation.py
    - backend/shap_explainability.py
    Επιστρέφει μήνυμα επιτυχίας.
    """
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".xlsx", ".csv"}:
        raise HTTPException(status_code=400, detail="Υποστήριξη μόνο για .xlsx ή .csv αρχεία.")

    # Αποθήκευση ως data.xlsx ή data.csv στο root
    target_path = BASE_DIR / f"data{suffix}"
    try:
        contents = file.file.read()
        target_path.write_bytes(contents)
    finally:
        file.file.close()

    # Τρέχουμε τη pipeline με subprocess, με cwd=BASE_DIR
    commands = [
        [sys.executable, "backend/preprocessing.py"],
        [sys.executable, "backend/training.py"],
        [sys.executable, "backend/evaluation.py"],
        [sys.executable, "backend/shap_explainability.py"],
    ]

    for cmd in commands:
        try:
            # Δεν κάνουμε text/capture_output για να μην έχουμε UnicodeDecodeError
            subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": f"Αποτυχία εντολής: {' '.join(cmd)}",
                    "returncode": exc.returncode,
                },
            )

    # Μετά την ολοκλήρωση της pipeline, ξαναφορτώνουμε τα μοντέλα στη μνήμη
    _load_models_and_info(force_reload=True)

    return {
        "status": "ok",
        "message": f"Το αρχείο '{filename}' ανέβηκε και η pipeline ολοκληρώθηκε.",
    }


def _register_insert(conn, req, password_hash, created_at, agreement_int):
    """Single attempt to insert a user. Raises on IntegrityError or OperationalError."""
    conn.execute(
        """
        INSERT INTO users (first_name, last_name, username, email, password_hash, agreement, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (req.firstName.strip(), req.lastName.strip(), req.username.strip(), req.email.strip(), password_hash, agreement_int, created_at),
    )
    conn.commit()


@app.post("/register")
def register(req: RegisterRequest):
    """
    Create account: save first name, last name, username, email, hashed password
    and agreement to SQLite (thelag.db). Retries on 'database is locked'.
    """
    password_hash = _hash_password(req.password)
    created_at = datetime.utcnow().isoformat() + "Z"
    agreement_int = 1 if req.agreement else 0

    max_attempts = 5
    for attempt in range(max_attempts):
        conn = _get_db_connection()
        try:
            _register_insert(conn, req, password_hash, created_at, agreement_int)
            return {"status": "ok", "message": "Account created successfully."}
        except sqlite3.IntegrityError as e:
            err_msg = str(e).lower()
            if "users.email" in err_msg or ".email" in err_msg:
                raise HTTPException(status_code=400, detail="Email already exists.")
            if "users.username" in err_msg or ".username" in err_msg:
                raise HTTPException(status_code=400, detail="Username already exists.")
            raise HTTPException(status_code=400, detail="Username or email already exists.")
        except sqlite3.OperationalError as e:
            err_str = str(e).lower()
            if "locked" in err_str or "busy" in err_str:
                if attempt < max_attempts - 1:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise HTTPException(
                    status_code=503,
                    detail="Database is busy. Close DB Browser for SQLite if it is open, then try again.",
                )
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try:
                conn.close()
            except Exception:
                pass
    raise HTTPException(
        status_code=503,
        detail="Database is busy. Close DB Browser for SQLite if it is open, then try again.",
    )


@app.post("/login")
@app.post("/login/")  # allow trailing slash so both URLs work
def login(req: LoginRequest):
    """
    Validate username and password against the database.
    Returns user info (username, firstName) on success so the frontend can show "Welcome FirstName!".
    """
    username = req.username.strip()
    password_hash = _hash_password(req.password)

    conn = _get_db_connection()
    try:
        row = conn.execute(
            "SELECT username, first_name, password_hash FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    stored_username, first_name, stored_hash = row
    if stored_hash != password_hash:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    return {
        "status": "ok",
        "message": "Logged in successfully.",
        "user": {"username": stored_username, "firstName": first_name or stored_username},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

