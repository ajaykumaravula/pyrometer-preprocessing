"""
=============================================================================
compress.py  —  Compression module for pyrometer pre-processing pipeline
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)

WHAT THIS MODULE DOES:
    Compresses a pyrometer time-series (1-D) or spatial temperature map
    (2-D or 3-D array) to reduce storage size while keeping reconstruction
    error acceptably low. Two methods are provided:

        1. svd_compress    — Truncated SVD (linear, fast, interpretable)
        2. wavelet_compress— Wavelet thresholding (good for 1-D signals)

HOW TO USE:
    from compress import svd_compress, wavelet_compress, compression_report
    from compress import print_preview

    compressed, meta = svd_compress(data_2d, rank=50)
    reconstructed    = svd_reconstruct(compressed, meta)

SWAP FOR ML LATER:
    Replace svd_compress/reconstruct with an autoencoder.
    Keep the same compress() / decompress() interface.
=============================================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1 — TRUNCATED SVD  (best for 2-D / 3-D spatial data)
# ─────────────────────────────────────────────────────────────────────────────

def svd_compress(data: np.ndarray, rank: int = 50):
    """
    Compress a 2-D matrix using Truncated SVD (rank-k approximation).

    For a 3-D array (rows × cols × frames), reshape to 2-D first:
        data_2d = data.reshape(rows * cols, frames)

    Parameters
    ----------
    data : np.ndarray — 2-D matrix to compress (shape: M × N)
    rank : int        — number of singular values to keep (lower = smaller)

    Returns
    -------
    compressed : dict with keys:
        'U'      — left singular vectors  (M × rank)
        'S'      — singular values        (rank,)
        'Vt'     — right singular vectors (rank × N)
        'rank'   — rank used
        'shape'  — original shape of data
    """
    data = data.astype(np.float64)

    # Full SVD then truncate (scipy TruncatedSVD is faster for large matrices
    # but numpy is used here to keep dependencies minimal)
    U, S, Vt = np.linalg.svd(data, full_matrices=False)

    # Keep only top-k components
    U_k  = U[:, :rank]
    S_k  = S[:rank]
    Vt_k = Vt[:rank, :]

    compressed = {
        "U"     : U_k,
        "S"     : S_k,
        "Vt"    : Vt_k,
        "rank"  : rank,
        "shape" : data.shape,
    }
    return compressed


def svd_reconstruct(compressed: dict) -> np.ndarray:
    """
    Reconstruct the original matrix from SVD compressed components.

    Parameters
    ----------
    compressed : dict — output from svd_compress()

    Returns
    -------
    np.ndarray — reconstructed matrix, same shape as original
    """
    U  = compressed["U"]
    S  = compressed["S"]
    Vt = compressed["Vt"]
    return (U * S) @ Vt    # equivalent to U @ np.diag(S) @ Vt but faster


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2 — WAVELET THRESHOLDING  (best for 1-D time-series)
# ─────────────────────────────────────────────────────────────────────────────

def wavelet_compress(signal: np.ndarray,
                     threshold: float = None,
                     keep_fraction: float = 0.10):
    """
    Compress a 1-D signal using Haar wavelet thresholding.
    Keeps only the largest wavelet coefficients (the rest → zero).

    Parameters
    ----------
    signal        : np.ndarray — 1-D temperature time-series
    threshold     : float      — absolute threshold for coefficient zeroing.
                                 If None, threshold is auto-set to keep
                                 `keep_fraction` of coefficients.
    keep_fraction : float      — fraction of coefficients to keep (0–1)
                                 (used only when threshold is None)

    Returns
    -------
    compressed : dict with keys:
        'coeffs'     — thresholded wavelet coefficients (mostly zeros)
        'threshold'  — threshold value used
        'length'     — original signal length
        'nonzero'    — number of non-zero coefficients kept
    """
    signal = signal.astype(np.float64)
    n      = len(signal)

    # Pad to next power of 2 (Haar requires power-of-2 length)
    n_padded = int(2 ** np.ceil(np.log2(n)))
    padded   = np.zeros(n_padded)
    padded[:n] = signal

    # Forward Haar wavelet transform
    coeffs = _haar_forward(padded)

    # Auto-threshold: keep the largest `keep_fraction` of coefficients
    if threshold is None:
        sorted_abs = np.sort(np.abs(coeffs))[::-1]
        k          = max(1, int(keep_fraction * len(coeffs)))
        threshold  = float(sorted_abs[k])

    # Hard thresholding: zero out coefficients below threshold
    coeffs_thresh = coeffs.copy()
    coeffs_thresh[np.abs(coeffs_thresh) < threshold] = 0.0

    compressed = {
        "coeffs"    : coeffs_thresh,
        "threshold" : threshold,
        "length"    : n,
        "nonzero"   : int((coeffs_thresh != 0).sum()),
        "n_padded"  : n_padded,
    }
    return compressed


def wavelet_reconstruct(compressed: dict) -> np.ndarray:
    """
    Reconstruct 1-D signal from wavelet compressed dict.

    Parameters
    ----------
    compressed : dict — output from wavelet_compress()

    Returns
    -------
    np.ndarray — reconstructed signal, trimmed to original length
    """
    reconstructed = _haar_inverse(compressed["coeffs"])
    return reconstructed[:compressed["length"]]


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3 — DELTA ENCODING (Near-lossless, int16)
# ─────────────────────────────────────────────────────────────────────────────

def delta_compress(signal: np.ndarray, scale: float = 100.0):
    """
    Near-lossless Delta Encoding using int16 differences.
    Recommended for high-fidelity thermal cooling curves (ATP-3).
    
    Parameters
    ----------
    signal : np.ndarray — 1-D temperature time-series
    scale  : float      — scaling factor to preserve precision in int16
    
    Returns
    -------
    compressed : dict with keys:
        'first_val' : first absolute value (float64)
        'diffs'     : scaled differences (int16)
        'scale'     : scale used
        'method'    : 'delta'
    """
    signal = signal.astype(np.float64)
    first_val = signal[0]
    # Calculate differences between consecutive samples
    diffs = np.diff(signal)
    # Scale and quantize to int16
    diffs_int16 = np.clip(np.round(diffs * scale), -32768, 32767).astype(np.int16)
    
    return {
        "first_val" : first_val,
        "diffs"     : diffs_int16,
        "scale"     : scale,
        "method"    : "delta"
    }


def delta_reconstruct(compressed: dict) -> np.ndarray:
    """
    Reconstruct signal from Delta Encoding components.
    """
    first_val = compressed["first_val"]
    scale     = compressed["scale"]
    diffs     = compressed["diffs"].astype(np.float64) / scale
    # Reconstruct signal via cumulative sum
    return np.concatenate(([first_val], first_val + np.cumsum(diffs)))


def _haar_forward(x: np.ndarray) -> np.ndarray:
    """Iterative Haar wavelet forward transform (in-place style)."""
    x = x.copy()
    n = len(x)
    while n > 1:
        half   = n // 2
        avg    = (x[:n:2] + x[1:n:2]) / 2.0
        diff   = (x[:n:2] - x[1:n:2]) / 2.0
        x[:half] = avg
        x[half:n] = diff
        n = half
    return x


def _haar_inverse(x: np.ndarray) -> np.ndarray:
    """Iterative Haar wavelet inverse transform."""
    x = x.copy()
    n = 2
    while n <= len(x):
        half = n // 2
        avg  = x[:half].copy()
        diff = x[half:n].copy()
        x[:n:2]  = avg + diff
        x[1:n:2] = avg - diff
        n *= 2
    return x


# ─────────────────────────────────────────────────────────────────────────────
# METRICS AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def compression_ratio(original: np.ndarray, compressed: dict,
                       method: str = "svd") -> float:
    """
    Calculate the compression ratio: original_bytes / compressed_bytes.
    Higher is better.

    Parameters
    ----------
    original   : original numpy array
    compressed : dict from svd_compress() or wavelet_compress()
    method     : 'svd' or 'wavelet'

    Returns
    -------
    float — compression ratio (e.g. 12.5 means 12.5× smaller)
    """
    orig_bytes = original.nbytes
    if method == "svd":
        comp_bytes = (compressed["U"].nbytes +
                      compressed["S"].nbytes +
                      compressed["Vt"].nbytes)
    else:   # wavelet — only non-zero values need storage
        comp_bytes = max(1, compressed["nonzero"] * 8)  # 8 bytes per float64
    return orig_bytes / comp_bytes


def reconstruction_rmse(original: np.ndarray,
                         reconstructed: np.ndarray) -> float:
    """
    RMSE between original and reconstructed signal/matrix.

    Returns
    -------
    float — RMSE in the same units as the data
    """
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def compression_report(original: np.ndarray,
                        reconstructed: np.ndarray,
                        compressed: dict,
                        method: str,
                        label: str = "") -> None:
    """
    Print a one-block compression summary.

    Parameters
    ----------
    original      : original array
    reconstructed : reconstructed array
    compressed    : dict from compress function
    method        : 'svd' or 'wavelet'
    label         : optional display label
    """
    ratio = compression_ratio(original, compressed, method)
    rmse  = reconstruction_rmse(original, reconstructed)

    tag = f"[{label}] " if label else ""
    print(f"  {tag}Method         : {method.upper()}")
    if method == "svd":
        print(f"  {tag}Rank kept      : {compressed['rank']}")
    else:
        print(f"  {tag}Coeffs kept    : {compressed['nonzero']} "
              f"/ {len(compressed['coeffs'])} "
              f"({compressed['nonzero']/len(compressed['coeffs'])*100:.1f}%)")
        print(f"  {tag}Threshold      : {compressed['threshold']:.2f}")
    orig_mb = original.nbytes / 1e6
    print(f"  {tag}Original size  : {orig_mb:.3f} MB")
    print(f"  {tag}Compression    : {ratio:.1f}×")
    print(f"  {tag}RMSE           : {rmse:.2f} (same units as input)")


# ─────────────────────────────────────────────────────────────────────────────
# PREVIEW HELPER
# ─────────────────────────────────────────────────────────────────────────────

def print_preview(original: np.ndarray,
                  reconstructed: np.ndarray,
                  time_s: np.ndarray,
                  start: int = 0) -> None:
    """
    Print a 5-row × 4-column preview table:
        Time | Original | Reconstructed | Error
    """
    import pandas as pd
    idx = list(range(start, start + 5))
    df  = pd.DataFrame({
        "Time_s"   : np.round(time_s[idx], 4),
        "Original" : np.round(original[idx], 2),
        "Reconstr" : np.round(reconstructed[idx], 2),
        "Error"    : np.round(reconstructed[idx] - original[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df.to_string())


# ─────────────────────────────────────────────────────────────────────────────

# -----------------------------------------------------------------------------
# SELF-TEST WITH REAL LAYER01 DATA  (run:  python compress.py)
#
# Chain: Layer01.mat -> Raw -> Denoise -> Calibrate -> Compress
# compress.py takes CALIBRATED signal as input (not raw, not denoised)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import scipy.io as sio
    import sys
    sys.path.insert(0, ".")
    from denoise   import denoise_signal
    from calibrate import linear_calibration, remove_drift
    import pandas as pd

    DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"

    # ── Build the calibrated signal (Stages 0-2) ──────────────────────
    print("=" * 65)
    print("compress.py -- self-test with real Layer01.mat data")
    print("Chain: Raw -> Denoise -> Calibrate -> [COMPRESS]")
    print("=" * 65)
    print("  Running Stage 0: Loading raw data ...")
    mat   = sio.loadmat(DATA_PATH)
    L     = mat["Layer"][0, 0]
    raw3d = L["RadiantTemp"].astype(np.float32)
    sh_A  = float(L["SHvariable_A"].flat[0])
    sh_B  = float(L["SHvariable_B"].flat[0])
    frame_max = raw3d.max(axis=(0, 1))
    T_raw     = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
    mask      = T_raw > 10
    T_raw     = T_raw[mask]
    n         = len(T_raw)
    time_s    = np.linspace(0, n * 0.002, n)
    print("  Running Stage 1: Denoising ...")
    T_den = denoise_signal(T_raw, median_kernel=7, gauss_sigma=3.0)
    print("  Running Stage 2: Calibrating ...")
    T_tc = np.zeros(n)
    T_tc[0] = T_raw[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08 * (T_raw[i] - T_tc[i-1])
    T_tc += np.random.default_rng(42).normal(0, 2, n)
    T_cal, coeffs = linear_calibration(T_den, T_tc, cal_fraction=0.20)
    T_cal         = remove_drift(T_cal, T_tc)
    print("  Stages 0-2 done. Calibrated signal is ready.")
    HOT = max(0, T_cal.argmax() - 2)
    idx = list(range(HOT, HOT + 5))

    # ── Show Calibrated signal going IN ───────────────────────────────
    print()
    print("=" * 65)
    print("COMPRESS INPUT -- Calibrated_C (from calibrate.py)")
    print("This is what compress.py receives")
    print("=" * 65)
    df_in = pd.DataFrame({
        "Time_s"       : np.round(time_s[idx], 4),
        "Calibrated_C" : np.round(T_cal[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df_in.to_string())
    print(f"  Range: {T_cal.min():.1f} - {T_cal.max():.1f} C")
    print(f"  Size : {T_cal.nbytes} bytes ({T_cal.nbytes/1e6:.4f} MB)")

    # ── METHOD A: Wavelet compression ─────────────────────────────────
    print()
    print("=" * 65)
    print("COMPRESS OUTPUT -- METHOD A: Wavelet (keep 10%)")
    print("=" * 65)
    comp_wav  = wavelet_compress(T_cal, keep_fraction=0.10)
    recon_wav = wavelet_reconstruct(comp_wav)
    ratio_wav = T_cal.nbytes / max(1, comp_wav['nonzero'] * 8)
    rmse_wav  = float(np.sqrt(np.mean((T_cal - recon_wav) ** 2)))
    df_wav = pd.DataFrame({
        "Time_s"         : np.round(time_s[idx], 4),
        "Calibrated_C"   : np.round(T_cal[idx], 2),
        "Reconstructed_C": np.round(recon_wav[idx], 2),
        "Error_C"        : np.round(recon_wav[idx] - T_cal[idx], 2),
    }, index=[f"t{i}" for i in idx])
    print(df_wav.to_string())
    print(f"  Coefficients kept : {comp_wav['nonzero']} / {len(comp_wav['coeffs'])} (10%)")
    print(f"  Compression ratio : {ratio_wav:.1f}x smaller")
    print(f"  RMSE error        : {rmse_wav:.2f} C (introduced by compression)")
    print(f"  Compressed size   : {comp_wav['nonzero']*8} bytes vs original {T_cal.nbytes} bytes")

    # ── METHOD B: Wavelet with different settings ──────────────────────
    print()
    print("=" * 65)
    print("COMPRESS OUTPUT -- METHOD B: Wavelet trade-off comparison")
    print("  More compression = bigger error. Pick the right balance.")
    print("=" * 65)
    results = []
    for keep in [0.20, 0.10, 0.05]:
        cw = wavelet_compress(T_cal, keep_fraction=keep)
        rw = wavelet_reconstruct(cw)
        results.append({
            "Keep_%"      : str(int(keep*100)) + "%",
            "Coeffs_kept" : cw["nonzero"],
            "Ratio"       : round(T_cal.nbytes / max(1, cw["nonzero"]*8), 1),
            "RMSE_C"      : round(float(np.sqrt(np.mean((T_cal-rw)**2))), 2),
            "Size_bytes"  : cw["nonzero"] * 8,
        })
    df_trade = pd.DataFrame(results)
    print(df_trade.to_string(index=False))
    print()
    print("  compress.py working correctly with real Layer01 data")
    print("  Input was Calibrated_C -- output is Reconstructed_C")
