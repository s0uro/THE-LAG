import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# === ΡΥΘΜΙΣΕΙΣ / ΟΝΟΜΑΤΑ ΣΤΗΛΩΝ ===
# Project root (parent of backend/). Paths resolved from here so they work when cwd differs (e.g. on server).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Raw file: use env THELAG_RAW_FILE if set (upload-and-run passes this), else data.xlsx / data.csv in project root.
_raw_env = os.environ.get("THELAG_RAW_FILE")
if _raw_env:
    RAW_FILE = _raw_env
else:
    _xlsx = _PROJECT_ROOT / "data.xlsx"
    _csv = _PROJECT_ROOT / "data.csv"
    RAW_FILE = str(_xlsx) if _xlsx.exists() else str(_csv)
OUTPUT_FILE = str(_PROJECT_ROOT / "backend" / "data_clean.csv")

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
    Υπολογίζει το Temporal Lag = dt(FT) - TRT(ET) για κάθε token.

    Το dataset έχει δύο τύπους γραμμών:
      - ET (idDevice='ET'): περιέχει TRT, FFD, FPD κ.λπ.
      - FT (idDevice=1):   περιέχει dt (χρόνος δακτύλου ανά token)

    Κάνουμε merge ανά (idUser, gid, tid) και υπολογίζουμε:
      Lag = dt_FT - TRT_ET

    Θετικό lag → το δάκτυλο προηγείται (εύκολη λέξη)
    Αρνητικό lag → το μάτι αργεί, το δάκτυλο μένει πίσω (δύσκολη λέξη)
    """
    # Case A: Ρητές στήλες Finger_Time / Eye_Time
    if COL_FINGER_TIME in df.columns and COL_EYE_TIME in df.columns:
        df[COL_FINGER_TIME] = pd.to_numeric(df[COL_FINGER_TIME], errors="coerce")
        df[COL_EYE_TIME] = pd.to_numeric(df[COL_EYE_TIME], errors="coerce")
        df["Lag"] = df[COL_FINGER_TIME] - df[COL_EYE_TIME]
        return df

    # Case B: Το dataset έχει στήλη idDevice που ξεχωρίζει ET από FT
    # Αυτή είναι η κύρια περίπτωση για το dedomena_ptixiakis.xlsx
    if "idDevice" in df.columns and COL_DT in df.columns and COL_TRT in df.columns:
        MERGE_KEYS = [c for c in ["idUser", "gid", "tid"] if c in df.columns]

        if len(MERGE_KEYS) >= 2:
            # Διαχωρισμός ET και FT
            et = df[df["idDevice"] == "ET"].copy()
            ft = df[df["idDevice"] == 1].copy()

            if et.empty or ft.empty:
                # Fallback αν ένα από τα δύο είναι κενό
                pass
            else:
                et["TRT_et"] = pd.to_numeric(et[COL_TRT], errors="coerce")
                # Κρατάμε μόνο TRT < 5s (αφαίρεση outliers)
                et["TRT_et"] = et["TRT_et"].where(et["TRT_et"] < 5)

                ft["dt_ft"] = pd.to_numeric(ft[COL_DT], errors="coerce")

                # Merge ET+FT ανά χρήστη + token
                merged = pd.merge(
                    et[MERGE_KEYS + ["TRT_et"]],
                    ft[MERGE_KEYS + ["dt_ft"]],
                    on=MERGE_KEYS,
                    how="inner",
                )
                merged["Lag"] = merged["dt_ft"] - merged["TRT_et"]

                # Αφαίρεση ακραίων τιμών lag (> 5s σε απόλυτη τιμή)
                merged = merged[merged["Lag"].abs() < 5]

                # Επιστρέφουμε μόνο τις ET γραμμές εμπλουτισμένες με Lag
                et_with_lag = pd.merge(
                    et,
                    merged[MERGE_KEYS + ["Lag"]],
                    on=MERGE_KEYS,
                    how="left",
                )
                print(
                    f"  Temporal Lag υπολογίστηκε για {merged['Lag'].notna().sum()} tokens "
                    f"(median={merged['Lag'].median():.3f}s)"
                )
                return et_with_lag

    # Case C: Fallback — χρησιμοποιούμε dt ως proxy για Lag
    if COL_DT in df.columns:
        df[COL_DT] = pd.to_numeric(df[COL_DT], errors="coerce")
        df["Lag"] = df[COL_DT]
        return df

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
    Δημιουργεί binary target για δυσκολία ανάγνωσης.

    Προτεραιότητα:
    1. Αν υπάρχει στήλη 'Lag' (Temporal Lag = dt_FT - TRT_ET):
       target = 1 αν Lag > median(Lag), αλλιώς 0
       (θετικό lag = εύκολη λέξη, αρνητικό = δύσκολη)

    2. Αν υπάρχουν TRT / isReg (παλιός ορισμός):
       target = 1 αν TRT > threshold ΚΑΙ isReg = 1

    3. Fallback: median split από διαθέσιμη αριθμητική στήλη
    """
    # Περίπτωση 1 (κύρια): Temporal Lag από merge ET+FT
    if "Lag" in df.columns:
        lag_series = pd.to_numeric(df["Lag"], errors="coerce")
        valid_lag = lag_series.dropna()
        if not valid_lag.empty:
            median_lag = valid_lag.median()
            # target=1 αν lag > median (εύκολη), target=0 αν lag <= median (δύσκολη)
            df["target"] = np.where(lag_series > median_lag, 1, 0)
            print(
                f"  Target από Temporal Lag (median={median_lag:.3f}s): "
                f"{(df['target']==1).sum()} εύκολες / {(df['target']==0).sum()} δύσκολες"
            )
            return df

    # Περίπτωση 2: TRT / isReg
    if COL_TRT in df.columns and COL_IS_REG in df.columns:
        if trt_threshold is None:
            trt_threshold = df[COL_TRT].mean()
        df["target"] = np.where(
            (df[COL_TRT] > trt_threshold) & (df[COL_IS_REG] == 1),
            1,
            0,
        )
        return df

    # Περίπτωση 3: Fallback median split
    alt_col = None
    for candidate in ["WAIS Digit Span Total", "WAIS Vocabulary", "freq", "len"]:
        if candidate in df.columns:
            alt_col = candidate
            break

    if alt_col is None:
        raise KeyError(
            "Δεν βρέθηκαν κατάλληλες στήλες για δημιουργία target. "
            "Χρειάζομαι 'Lag', 'TRT'+'isReg', ή κάποια από: "
            "'WAIS Digit Span Total', 'WAIS Vocabulary', 'freq', 'len'."
        )

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

