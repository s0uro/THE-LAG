import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# === ΡΥΘΜΙΣΕΙΣ / ΟΝΟΜΑΤΑ ΣΤΗΛΩΝ ===
# Διαβάζουμε το βασικό αρχείο `data.xlsx` από το root του project.
# Αν θέλεις άλλο path, άλλαξέ το εδώ.
RAW_FILE = "data.xlsx"  # ή "data.csv" αν έχεις CSV
OUTPUT_FILE = "backend/data_clean.csv"

# Ονόματα στηλών (προσάρμοσέ τα αν διαφέρουν)
# Στο δικό σου αρχείο ΔΕΝ υπάρχουν Finger_Time/Eye_Time. Συνήθως υπάρχει ήδη διαφορά χρόνου ως `dt`.
# Αν παρ' όλα αυτά έχεις άλλες στήλες για finger/eye time, βάλε τα ονόματα εδώ.
COL_FINGER_TIME = "Finger_Time"
COL_EYE_TIME = "Eye_Time"
COL_DT = "dt"  # fallback για Lag αν λείπουν οι παραπάνω
COL_TRT = "TRT"
COL_IS_REG = "isReg"
COL_LEN = "len"
COL_FREQ = "freq"
COL_FFD = "FFD"
COL_FPD = "FPD"

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
        return pd.read_csv(path)
    else:
        raise ValueError(f"Μη υποστηριζόμενο format αρχείου: {ext}")


def replace_scientific_outliers_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Εντοπίζει πολύ μεγάλες τιμές (π.χ. > 1e10) σε όλες τις αριθμητικές στήλες
    και τις αντικαθιστά με τη διάμεσο της κάθε στήλης.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_values = df[col]
        # Θεωρούμε outliers ό,τι είναι πιο μεγάλο από 1e10 (π.χ. 1e14)
        mask_outliers = col_values.abs() > 1e10
        if mask_outliers.any():
            median_val = col_values[~mask_outliers].median()
            df.loc[mask_outliers, col] = median_val

    return df


def fill_missing_ffd_fpd_trt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Συμπλήρωση κενών (NA) στις FFD, FPD, TRT με γραμμική παρεμβολή
    κατά index και, αν μείνουν κενά, με το μέσο όρο της στήλης.
    """
    for col in [COL_FFD, COL_FPD, COL_TRT]:
        if col not in df.columns:
            continue

        # Μετατροπή σε αριθμητικό τύπο (μη-αριθμητικά -> NaN)
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Γραμμική παρεμβολή
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

        # Αν τυχόν μείνουν NaN (π.χ. στην αρχή/τέλος), χρησιμοποίησε μέσο όρο
        if df[col].isna().any():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

    return df


def add_lag_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Προσθέτει στήλη Lag = Finger_Time - Eye_Time.
    """
    # Case A: Έχουμε και τις δύο στήλες finger/eye time
    if COL_FINGER_TIME in df.columns and COL_EYE_TIME in df.columns:
        df[COL_FINGER_TIME] = pd.to_numeric(df[COL_FINGER_TIME], errors="coerce")
        df[COL_EYE_TIME] = pd.to_numeric(df[COL_EYE_TIME], errors="coerce")
        df["Lag"] = df[COL_FINGER_TIME] - df[COL_EYE_TIME]
        return df

    # Case B: Στο dataset υπάρχει ήδη dt (διαφορά χρόνου)
    if COL_DT in df.columns:
        df[COL_DT] = pd.to_numeric(df[COL_DT], errors="coerce")
        df["Lag"] = df[COL_DT]
        return df

    # Case C: Δεν υπάρχουν καθόλου σχετικές στήλες -> δεν υπολογίζουμε Lag,
    # απλώς συνεχίζουμε χωρίς Lag για να μην "σπάει" ο κώδικας.
    # Αν αργότερα προσθέσεις τέτοιες στήλες, αυτή η συνάρτηση θα τις αξιοποιήσει.
    return df


def normalize_len_freq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Κλίμακα 0-1 (MinMax) για len και freq, ώστε το MLP να μην επηρεάζεται
    από μεγάλες διαφορές τιμών.
    """
    cols_to_scale = [c for c in [COL_LEN, COL_FREQ] if c in df.columns]
    if not cols_to_scale:
        return df

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols_to_scale])
    df[[f"{c}_scaled" for c in cols_to_scale]] = scaled
    return df


def create_target_label(df: pd.DataFrame, trt_threshold: float | None = None) -> pd.DataFrame:
    """
    Δημιουργεί binary target:
    - target = 1 (Δύσκολη λέξη) αν TRT > threshold και isReg = 1
    - target = 0 αλλιώς

    Αν δεν δοθεί threshold, χρησιμοποιεί το μέσο όρο της TRT.
    """
    # Περίπτωση 1: Έχουμε TRT / isReg -> χρησιμοποιούμε τον αρχικό ορισμό
    if COL_TRT in df.columns and COL_IS_REG in df.columns:
        if trt_threshold is None:
            trt_threshold = df[COL_TRT].mean()

        df["target"] = np.where(
            (df[COL_TRT] > trt_threshold) & (df[COL_IS_REG] == 1),
            1,
            0,
        )
        return df

    # Περίπτωση 2: Δεν υπάρχουν TRT / isReg.
    # Τότε δημιουργούμε ένα binary target από κάποια υπάρχουσα στήλη,
    # π.χ. από το "WAIS Digit Span Total" αν είναι διαθέσιμη:
    alt_col = None
    for candidate in [
        "WAIS Digit Span Total",
        "WAIS Vocabulary",
        "freq",
        "len",
    ]:
        if candidate in df.columns:
            alt_col = candidate
            break

    if alt_col is None:
        raise KeyError(
            "Δεν βρέθηκαν οι στήλες TRT/isReg ούτε κάποια εναλλακτική "
            "στήλη (π.χ. 'WAIS Digit Span Total', 'WAIS Vocabulary', 'freq', 'len') "
            "για δημιουργία binary target."
        )

    # Χρησιμοποιούμε το μέσο όρο της alt_col ως όριο: πάνω από το mean -> target=1, αλλιώς 0
    threshold = df[alt_col].mean()
    df["target"] = np.where(df[alt_col] > threshold, 1, 0)
    return df


def main() -> None:
    _force_utf8_stdio()
    print("Διαβάζω αρχείο δεδομένων...")
    df = read_input(RAW_FILE)
    print(f"Αρχικό σχήμα: {df.shape}")

    # 1. Καθαρισμός πολύ μεγάλων τιμών (π.χ. e+14)
    df = replace_scientific_outliers_with_median(df)

    # 2. Συμπλήρωση κενών σε FFD, FPD, TRT
    df = fill_missing_ffd_fpd_trt(df)

    # 3. Υπολογισμός Lag
    df = add_lag_column(df)

    # 4. Κανονικοποίηση len και freq σε [0, 1]
    df = normalize_len_freq(df)

    # 5. Δημιουργία target label (Difficulty)
    df = create_target_label(df, trt_threshold=None)

    # 6. Αποθήκευση καθαρισμένου αρχείου
    # UTF-8 output avoids Windows codepage issues when data contains Greek.
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Αποθήκευσα το καθαρισμένο αρχείο στο '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()

