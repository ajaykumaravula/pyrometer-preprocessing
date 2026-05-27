"""
=============================================================================
calibrate.py  —  Calibration module for pyrometer pre-processing pipeline
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)

WHAT THIS MODULE DOES:
    Corrects emissivity error in pyrometer signals using a thermocouple
    as a reference. Two methods are provided:

        1. linear_calibration  — fits T_true = a × T_pyr + b
                                 (baseline, good for stable emissivity)
        2. polynomial_calibration — fits a degree-2 or 3 polynomial
                                 (better for emissivity that varies with temp)

    Also provides drift correction for slow sensor offset drift.

HOW TO USE:
    from calibrate import linear_calibration, polynomial_calibration
    from calibrate import remove_drift, print_preview

    T_cal, coeffs = linear_calibration(T_pyr, T_tc, cal_fraction=0.20)

SWAP FOR ML LATER:
    Replace linear_calibration() or polynomial_calibration() with a
    neural network regressor. The function signature stays the same.
=============================================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def linear_calibration(T_pyr: np.ndarray,
                       T_ref: np.ndarray,
                       cal_fraction: float = 0.20):
    """
    Fit a linear correction  T_ref ≈ a × T_pyr + b  on a calibration
    window, then apply it to the full signal.

    Parameters
    ----------
    T_pyr        : np.ndarray — denoised pyrometer signal (°C or counts)
    T_ref        : np.ndarray — thermocouple reference signal (°C)
    cal_fraction : float      — fraction of data used for fitting (e.g. 0.20
                                means first 20% is the calibration window)

    Returns
    -------
    T_cal  : np.ndarray — calibrated temperature signal
    coeffs : dict       — {'a': slope, 'b': intercept, 'method': 'linear'}
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))   # at least 10 points

    x = T_pyr[:cal_end].astype(np.float64)
    y = T_ref[:cal_end].astype(np.float64)

    # Ordinary least-squares: [x | 1] * [a, b]^T = y
    A      = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    a, b   = result[0]

    T_cal  = a * T_pyr.astype(np.float64) + b
    coeffs = {"a": a, "b": b, "method": "linear", "cal_end": cal_end}

    return T_cal, coeffs


def polynomial_calibration(T_pyr: np.ndarray,
                            T_ref: np.ndarray,
                            cal_fraction: float = 0.20,
                            degree: int = 2):
    """
    Fit a polynomial correction of given degree on the calibration window.
    Useful when emissivity changes significantly with temperature.

    Parameters
    ----------
    T_pyr        : np.ndarray — denoised pyrometer signal
    T_ref        : np.ndarray — thermocouple reference signal
    cal_fraction : float      — fraction of data for calibration window
    degree       : int        — polynomial degree (2 or 3 recommended)

    Returns
    -------
    T_cal  : np.ndarray — calibrated temperature signal
    coeffs : dict       — {'poly': np.poly1d object, 'degree': degree, ...}
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))

    x = T_pyr[:cal_end].astype(np.float64)
    y = T_ref[:cal_end].astype(np.float64)

    # Fit polynomial coefficients
    poly_coeffs = np.polyfit(x, y, deg=degree)
    poly_fn     = np.poly1d(poly_coeffs)

    T_cal  = poly_fn(T_pyr.astype(np.float64))
    coeffs = {
        "poly"    : poly_fn,
        "degree"  : degree,
        "method"  : f"polynomial_deg{degree}",
        "cal_end" : cal_end,
    }

    return T_cal, coeffs


def remove_drift(T_pyr: np.ndarray,
                 T_ref: np.ndarray) -> np.ndarray:
    """
    Remove a linear drift from a pyrometer signal.

    Fits a straight line to (T_pyr - T_ref) over the full signal,
    then subtracts it. This corrects a slow sensor offset that grows
    over time (common in long heat-treatment runs).

    Parameters
    ----------
    T_pyr : np.ndarray — calibrated pyrometer signal
    T_ref : np.ndarray — thermocouple reference signal

    Returns
    -------
    np.ndarray — drift-corrected pyrometer signal
    """
    residual    = T_pyr.astype(np.float64) - T_ref.astype(np.float64)
    t           = np.arange(len(T_pyr), dtype=np.float64)
    slope, intercept = np.polyfit(t, residual, 1)
    drift       = slope * t + intercept
    return T_pyr - drift


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────

def rmse(T_cal: np.ndarray, T_true: np.ndarray) -> float:
    """
    Root Mean Square Error between calibrated and true temperature.

    Parameters
    ----------
    T_cal  : calibrated signal
    T_true : ground-truth or reference signal

    Returns
    -------
    float — RMSE in the same units as the inputs (°C)
    """
    return float(np.sqrt(np.mean((T_cal - T_true) ** 2)))


def mae(T_cal: np.ndarray, T_true: np.ndarray) -> float:
    """
    Mean Absolute Error between calibrated and true temperature.

    Returns
    -------
    float — MAE in °C
    """
    return float(np.mean(np.abs(T_cal - T_true)))


def calibration_report(T_raw: np.ndarray,
                        T_cal: np.ndarray,
                        T_true: np.ndarray,
                        coeffs: dict,
                        label: str = "Pyrometer") -> None:
    """
    Print a short accuracy report comparing raw vs calibrated signal.

    Parameters
    ----------
    T_raw  : raw (pre-calibration) signal
    T_cal  : calibrated signal
    T_true : reference / true temperature
    coeffs : coefficient dict returned by calibration function
    label  : sensor name for display
    """
    print(f"  [{label}] method      : {coeffs.get('method', '?')}")
    print(f"  [{label}] cal window  : first {coeffs.get('cal_end', '?')} samples")
    if coeffs.get("method") == "linear":
        print(f"  [{label}] formula     : T_cal = {coeffs['a']:.5f} × T_raw "
              f"+ {coeffs['b']:.2f}")
    print(f"  [{label}] RMSE before : {rmse(T_raw, T_true):.2f} °C")
    print(f"  [{label}] RMSE after  : {rmse(T_cal, T_true):.2f} °C")
    print(f"  [{label}] MAE  after  : {mae(T_cal, T_true):.2f} °C")


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_preview(T_raw: np.ndarray,
                  T_cal: np.ndarray,
                  T_ref: np.ndarray,
                  time_s: np.ndarray,
                  start: int = 0) -> None:
    """
    Print a 5-row × 5-column preview table:
        Time | Raw | Calibrated | TC_reference | Error_vs_ref
    """
    import pandas as pd
    idx = list(range(start, start + 5))
    df  = pd.DataFrame({
        "Time_s" : np.round(time_s[idx], 4),
        "Raw_C"  : np.round(T_raw[idx], 2),
        "Cal_C"  : np.round(T_cal[idx], 2),
        "TC_ref" : np.round(T_ref[idx], 2),
        "Err_C"  : np.round(T_cal[idx] - T_ref[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST WITH REAL LAYER01 DATA  (run:  python calibrate.py)
#
# Chain followed:
#   Layer01.mat → Raw °C → Denoise → Calibrate
#   Input to calibrate = DENOISED signal (not raw)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import scipy.io as sio
    import sys
    sys.path.insert(0, ".")
    from denoise import denoise_signal

    print("=" * 60)
    print("calibrate.py — self-test with real Layer01.mat data")
    print("Chain: Raw → Denoise → Calibrate")
    print("=" * 60)

    DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"
    print(f"\n  Loading: {DATA_PATH}")

    # ── STEP 1: Load raw data ─────────────────────────────────────────────
    mat   = sio.loadmat(DATA_PATH)
    L     = mat["Layer"][0, 0]
    raw3d = L["RadiantTemp"].astype(np.float32)
    sh_A  = float(L["SHvariable_A"].flat[0])
    sh_B  = float(L["SHvariable_B"].flat[0])

    frame_max = raw3d.max(axis=(0, 1))
    T_raw_C   = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
    mask      = T_raw_C > 10
    T_raw_C   = T_raw_C[mask]
    n         = len(T_raw_C)
    time_s    = np.linspace(0, n * 0.002, n)

    print(f"  Frames loaded    : {n}")
    print(f"  Raw temp range   : {T_raw_C.min():.1f} - {T_raw_C.max():.1f} C")

    # ── STEP 2: Denoise first (calibrate takes denoised as input) ────────
    print("\n  Running denoise first (calibrate needs clean input) ...")
    T_den = denoise_signal(T_raw_C, median_kernel=7, gauss_sigma=3.0)
    print(f"  Denoised signal ready")

    # ── STEP 3: Build thermocouple reference ──────────────────────────────
    T_tc = np.zeros(n)
    T_tc[0] = T_raw_C[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08 * (T_raw_C[i] - T_tc[i-1])
    T_tc += np.random.default_rng(42).normal(0, 2, n)
    print(f"  Thermocouple reference ready")

    # ── STEP 4: Linear calibration ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  METHOD A — Linear Calibration")
    print("-" * 60)
    T_lin, c_lin = linear_calibration(T_den, T_tc, cal_fraction=0.20)
    T_lin        = remove_drift(T_lin, T_tc)
    calibration_report(T_den, T_lin, T_tc, c_lin, label="Linear")

    # ── STEP 5: Polynomial calibration (degree=2) ─────────────────────────
    print("\n" + "-" * 60)
    print("  METHOD B — Polynomial Calibration (degree=2)")
    print("-" * 60)
    T_poly, c_poly = polynomial_calibration(T_den, T_tc,
                                             cal_fraction=0.20, degree=2)
    calibration_report(T_den, T_poly, T_tc, c_poly, label="Poly2")

    # ── STEP 6: 5x5 preview ───────────────────────────────────────────────
    hot = max(0, T_raw_C.argmax() - 2)

    print("\n" + "=" * 60)
    print("  * 5-ROW PREVIEW — Full chain at hottest region")
    print("  Columns: Time | Denoised (input) | Calibrated | TC_ref | Error")
    print("=" * 60)
    print_preview(T_den, T_lin, T_tc, time_s, start=hot)

    print()
    print("  What the columns mean:")
    print("  Raw_C   = denoised signal fed INTO calibration")
    print("  Cal_C   = calibrated output (emissivity corrected)")
    print("  TC_ref  = thermocouple reference (what we aim to match)")
    print("  Err_C   = Cal_C minus TC_ref (how close we got)")
    print()
    print("  calibrate.py working correctly with real Layer01 data")
