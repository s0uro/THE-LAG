import json

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import correlate


INPUT_FILE = "backend/data_clean.csv"  # έξοδος από το preprocessing
LAG_STATS_JSON_PATH = "lag_stats.json"

COL_FINGER_TIME = "Finger_Time"
COL_EYE_TIME = "Eye_Time"
COL_DT = "dt"
COL_LAG = "Lag"
COL_COVERAGE = "coverage"
COL_TRT = "TRT"

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


def compute_cross_correlation_and_lag(
    finger_signal: np.ndarray, eye_signal: np.ndarray, sample_rate_hz: float | None = None
) -> tuple[float, int]:
    """
    Υπολογίζει τη διασταυρούμενη συσχέτιση (cross-correlation) μεταξύ
    finger_signal και eye_signal, και επιστρέφει:
    - τη μέγιστη τιμή συσχέτισης
    - το lag (σε δείγματα) όπου αυτή η συσχέτιση είναι μέγιστη

    Αν δοθεί sample_rate_hz, μπορείς να μετατρέψεις το lag σε ms:
        lag_seconds = lag_samples / sample_rate_hz
        lag_ms = lag_seconds * 1000
    """
    finger_signal = np.asarray(finger_signal)
    eye_signal = np.asarray(eye_signal)

    # Αφαιρούμε το μέσο για να εστιάσει η συσχέτιση στο σχήμα του σήματος
    finger_centered = finger_signal - finger_signal.mean()
    eye_centered = eye_signal - eye_signal.mean()

    corr = correlate(finger_centered, eye_centered, mode="full")
    lags = np.arange(-len(eye_centered) + 1, len(finger_centered))

    max_idx = np.argmax(corr)
    max_corr = corr[max_idx]
    best_lag = lags[max_idx]

    return float(max_corr), int(best_lag)


def main() -> None:
    _force_utf8_stdio()
    print(f"Διαβάζω προεπεξεργασμένα δεδομένα από '{INPUT_FILE}'...")
    df = read_input(INPUT_FILE)

    # Case A: Αν υπάρχουν 2 σήματα (Finger_Time/Eye_Time), κάνε cross-correlation κανονικά
    if COL_FINGER_TIME in df.columns and COL_EYE_TIME in df.columns:
        finger_signal = pd.to_numeric(df[COL_FINGER_TIME], errors="coerce").fillna(0).values
        eye_signal = pd.to_numeric(df[COL_EYE_TIME], errors="coerce").fillna(0).values

        max_corr, best_lag_samples = compute_cross_correlation_and_lag(
            finger_signal, eye_signal, sample_rate_hz=None
        )

        print("=== Cross-Correlation Αποτελέσματα ===")
        print(f"Μέγιστη τιμή συσχέτισης      : {max_corr:.4f}")
        print(f"Lag (σε δείγματα/index)     : {best_lag_samples}")
        print(
            "Ερμηνεία: θετικό lag σημαίνει ότι το Finger_Time 'ακολουθεί' το Eye_Time,\n"
            "ενώ αρνητικό lag ότι το Finger_Time 'προηγείται'."
        )
        return

    # Case B: Αν δεν έχεις 2 σήματα, αλλά έχεις ήδη Lag/dt, βγάλε περιγραφικά στατιστικά
    if COL_LAG in df.columns:
        lag_series = pd.to_numeric(df[COL_LAG], errors="coerce")
        print("=== Lag (από στήλη 'Lag') ===")
        print(f"Count : {lag_series.notna().sum()}")
        print(f"Mean  : {lag_series.mean():.4f}")
        print(f"Median: {lag_series.median():.4f}")
        print(f"Std   : {lag_series.std():.4f}")

        # Υπολογισμός histogram για frontend visualization
        clean = lag_series.dropna()
        if not clean.empty:
            # Πετάμε ακραία outliers (1ο–99ο percentile) ώστε το histogram να είναι πιο “συμπαγές”
            low, high = np.percentile(clean, [1, 99])
            filtered = clean[(clean >= low) & (clean <= high)]
            counts, bin_edges = np.histogram(filtered, bins=30)
            lag_stats = {
                "count": int(filtered.shape[0]),
                "mean": float(filtered.mean()),
                "median": float(filtered.median()),
                "std": float(filtered.std()),
                "min": float(filtered.min()),
                "max": float(filtered.max()),
                "histogram": {
                    "bins": bin_edges.tolist(),
                    "counts": counts.tolist(),
                },
            }
            with open(LAG_STATS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(lag_stats, f, indent=2, ensure_ascii=False)
            print(f"Αποθήκευσα lag histogram/statistics στο '{LAG_STATS_JSON_PATH}'")
        return

    if COL_DT in df.columns:
        dt_series = pd.to_numeric(df[COL_DT], errors="coerce")
        print("=== Lag (fallback από στήλη 'dt') ===")
        print(f"Count : {dt_series.notna().sum()}")
        print(f"Mean  : {dt_series.mean():.4f}")
        print(f"Median: {dt_series.median():.4f}")
        print(f"Std   : {dt_series.std():.4f}")
        return

    # Case C: Τελευταίο fallback (αν θες να το χρησιμοποιήσεις ως proxy-signals)
    if COL_COVERAGE in df.columns and COL_TRT in df.columns:
        finger_signal = pd.to_numeric(df[COL_COVERAGE], errors="coerce").fillna(0).values
        eye_signal = pd.to_numeric(df[COL_TRT], errors="coerce").fillna(0).values
        max_corr, best_lag_samples = compute_cross_correlation_and_lag(
            finger_signal, eye_signal, sample_rate_hz=None
        )
        print("=== Cross-Correlation (proxy: coverage vs TRT) ===")
        print(f"Μέγιστη τιμή συσχέτισης      : {max_corr:.4f}")
        print(f"Lag (σε δείγματα/index)     : {best_lag_samples}")
        return

    raise KeyError(
        "Δεν βρήκα κατάλληλες στήλες για cross-correlation ή lag stats. "
        "Χρειάζομαι είτε (Finger_Time & Eye_Time) ή Lag/dt."
    )


if __name__ == "__main__":
    main()

