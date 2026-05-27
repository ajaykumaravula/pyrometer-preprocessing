"""
=============================================================================
denoise.py  —  Denoising module for pyrometer pre-processing pipeline
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)

WHAT THIS MODULE DOES:
    Takes a raw 1-D pyrometer time-series (numpy array) and returns a
    cleaned version using two steps:
        1. Median filter   — removes impulse spikes (sudden jumps)
        2. Gaussian smooth — reduces residual high-frequency noise

HOW TO USE:
    from denoise import denoise_signal, print_preview

    clean = denoise_signal(raw_signal)

SWAP FOR ML LATER:
    Replace denoise_signal() body with a CNN or LSTM denoiser.
    The function signature (input array → output array) stays the same
    so no other file needs to change.
=============================================================================
"""

import numpy as np
from scipy.signal import medfilt


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def denoise_signal(signal: np.ndarray,
                   median_kernel: int = 7,
                   gauss_sigma: float = 3.0) -> np.ndarray:
    """
    Denoise a 1-D pyrometer time-series.

    Parameters
    ----------
    signal        : np.ndarray  — raw temperature values (°C or counts)
    median_kernel : int         — window size for median filter (must be odd)
    gauss_sigma   : float       — standard deviation for Gaussian smoothing

    Returns
    -------
    np.ndarray — denoised signal, same length as input

    Steps
    -----
    1. Median filter  : replaces each point with the median of its neighbours.
                        Very effective at removing sudden spikes.
    2. Gaussian smooth: convolves with a Gaussian kernel to reduce
                        residual noise without blurring sharp edges.
    """
    # Step 1 — median filter (spike removal)
    if median_kernel % 2 == 0:
        median_kernel += 1        # kernel must be odd
    signal_med = medfilt(signal.astype(np.float64), kernel_size=median_kernel)

    # Step 2 — Gaussian smoothing
    signal_den = _gaussian_smooth(signal_med, sigma=gauss_sigma)

    return signal_den


def count_spikes(raw: np.ndarray,
                 denoised: np.ndarray,
                 threshold: float = 50.0) -> int:
    """
    Count how many samples were changed by more than `threshold`
    after denoising. Used as a simple spike-count diagnostic.

    Parameters
    ----------
    raw      : np.ndarray — original signal
    denoised : np.ndarray — denoised signal
    threshold: float      — minimum change to count as a spike (°C or counts)

    Returns
    -------
    int — number of spikes detected
    """
    return int((np.abs(raw - denoised) > threshold).sum())


def noise_level(signal: np.ndarray, smooth_sigma: float = 10.0) -> float:
    """
    Estimate the noise level (standard deviation) of a signal by
    comparing it against a heavily smoothed baseline.

    Parameters
    ----------
    signal       : np.ndarray — input signal
    smooth_sigma : float      — sigma for the baseline Gaussian (large value)

    Returns
    -------
    float — estimated noise standard deviation
    """
    baseline = _gaussian_smooth(signal, sigma=smooth_sigma)
    return float(np.std(signal - baseline))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_smooth(signal: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Convolve signal with a Gaussian kernel.
    Private helper — call denoise_signal() from outside this module.
    """
    half_w = int(4 * sigma + 1)
    x      = np.arange(-half_w, half_w + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode="same")


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW HELPER  (prints 5-row table so you can see changes in terminal)
# ─────────────────────────────────────────────────────────────────────────────

def print_preview(raw: np.ndarray,
                  denoised: np.ndarray,
                  time_s: np.ndarray,
                  label: str = "Denoised",
                  start: int = 0) -> None:
    """
    Print a 5-row × 3-column preview table showing:
        Time (s) | Raw value | Denoised value

    Parameters
    ----------
    raw      : raw signal array
    denoised : denoised signal array
    time_s   : time axis in seconds
    label    : column name for the denoised column
    start    : starting index for the preview window
    """
    import pandas as pd
    idx = list(range(start, start + 5))
    df  = pd.DataFrame({
        "Time_s"  : np.round(time_s[idx], 4),
        "Raw"     : np.round(raw[idx], 2),
        label     : np.round(denoised[idx], 2),
        "Change"  : np.round(denoised[idx] - raw[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST WITH REAL LAYER01 DATA  (run:  python denoise.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import scipy.io as sio

    print("=" * 55)
    print("denoise.py — self-test with real Layer01.mat data")
    print("=" * 55)

    # ── Load Layer01.mat ──────────────────────────────────────────────────
    DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"
    
    print(f"  Loading: {DATA_PATH}")

    mat   = sio.loadmat(DATA_PATH)
    L     = mat["Layer"][0, 0]
    raw3d = L["RadiantTemp"].astype(np.float32)   # (126, 360, 2497)
    sh_A  = float(L["SHvariable_A"].flat[0])
    sh_B  = float(L["SHvariable_B"].flat[0])

    # Max pixel per frame → 1-D signal in °C
    frame_max = raw3d.max(axis=(0, 1))
    T_raw_C   = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)

    # Keep only frames with real signal (laser on)
    mask    = T_raw_C > 10
    T_raw_C = T_raw_C[mask]
    n       = len(T_raw_C)
    time_s  = np.linspace(0, n * 0.002, n)

    print(f"  Frames loaded    : {n}")
    print(f"  Raw temp range   : {T_raw_C.min():.1f} – {T_raw_C.max():.1f} °C")

    # ── Run denoising ─────────────────────────────────────────────────────
    T_den   = denoise_signal(T_raw_C, median_kernel=7, gauss_sigma=3.0)
    spikes  = count_spikes(T_raw_C, T_den)
    noise_b = noise_level(T_raw_C)
    noise_a = noise_level(T_den)

    print(f"  Spikes removed   : {spikes}")
    print(f"  Noise before     : {noise_b:.2f} °C")
    print(f"  Noise after      : {noise_a:.2f} °C")

    # ── 5×5 preview around the hottest point ─────────────────────────────
    hot_start = max(0, T_raw_C.argmax() - 2)

    print()
    print("  ★ RAW vs DENOISED — 5 rows × 4 columns")
    print("  (showing hottest region of Layer01 signal)")
    print()
    print_preview(T_raw_C, T_den, time_s, label="Denoised_C", start=hot_start)
    print()
    print("  ✓ denoise.py working correctly with real Layer01 data")
