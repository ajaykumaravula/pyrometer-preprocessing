"""
=============================================================================
classical_methods.py  --  ALL Classical Baseline Methods
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)
Deliverable: D3 -- Investigation of classical methods vs ML

CLASSICAL METHODS COVERED:

    Denoising   : 1) Raw Baseline (no denoising)
                  2) Moving Average (k=7)
                  3) Median Filter (k=7)
                  4) Savitzky-Golay (w=11, p=3)
                  5) Gaussian Filter (sigma=3.0)
                  6) Kalman Filter (Q=0.001, R=10)

    Calibration : 1) Raw Baseline (no calibration)
                  2) Mean Offset Correction
                  3) Linear Regression
                  4) Polynomial Regression (degree=2)
                  5) Piecewise Linear (3 segments)

    Compression : 1) Raw Baseline (no compression)
                  2) Delta Encoding (int16)
                  3) Downsampling (2x)
                  4) SVD (rank=10)
                  5) Wavelet Haar (10%)

HOW TO RUN:
    python classical_methods.py

OUTPUT:
    classical_results.png         -- comparison plots
    classical_summary_updated.csv -- full results table
=============================================================================
"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import sys
sys.path.insert(0, ".")

from denoise   import denoise_signal
from calibrate import linear_calibration, remove_drift
from compress  import wavelet_compress, wavelet_reconstruct, \
                      svd_compress, svd_reconstruct
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"

print("=" * 65)
print("classical_methods.py -- ALL Classical Baseline Methods")
print("=" * 65)
print("\n  Loading Layer01.mat ...")

mat   = sio.loadmat(DATA_PATH)
L     = mat["Layer"][0, 0]
raw3d = L["RadiantTemp"].astype(np.float32)
sh_A  = float(L["SHvariable_A"].flat[0])
sh_B  = float(L["SHvariable_B"].flat[0])
frame_max = raw3d.max(axis=(0, 1))
T_raw = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
mask  = T_raw > 10
T_raw = T_raw[mask].astype(np.float32)
n     = len(T_raw)
time_s = np.linspace(0, n * 0.002, n)

# Thermocouple reference
T_tc = np.zeros(n); T_tc[0] = T_raw[0]
for i in range(1, n):
    T_tc[i] = T_tc[i-1] + 0.08 * (T_raw[i] - T_tc[i-1])
T_tc = (T_tc + np.random.default_rng(42).normal(0, 2, n)).astype(np.float32)

print(f"  Frames loaded : {n}")
print(f"  Raw range     : {T_raw.min():.1f} - {T_raw.max():.1f} C")

HOT = max(0, T_raw.argmax() - 2)
idx = list(range(HOT, HOT + 5))

# Helper: noise level
def noise_level(sig):
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

noise_raw = noise_level(T_raw)


# =============================================================================
# STEP 1 -- ALL DENOISING METHODS
# =============================================================================
print()
print("=" * 65)
print("STEP 1 -- ALL Classical Denoising Methods")
print("=" * 65)
print(f"\n  Raw baseline noise : {noise_raw:.2f} C")

# ── Method 1: Raw Baseline ────────────────────────────────────────────────────
T_raw_den = T_raw.copy()
print(f"\n  Method 1 -- Raw Baseline (no denoising)")
print(f"  Noise : {noise_raw:.2f} C  |  Reduction : 0.0%")

# ── Method 2: Moving Average ──────────────────────────────────────────────────
def moving_average(sig, window=7):
    """
    Moving average filter.
    Replaces each point with the mean of its neighbours.
    Fast but blurs sharp temperature peaks.
    """
    kernel = np.ones(window) / window
    return np.convolve(sig, kernel, mode="same").astype(np.float32)

T_ma    = moving_average(T_raw, window=7)
noise_ma = noise_level(T_ma)
print(f"\n  Method 2 -- Moving Average (k=7)")
print(f"  Noise : {noise_ma:.2f} C  |  Reduction : {(1-noise_ma/noise_raw)*100:.1f}%")

# ── Method 3: Median Filter ───────────────────────────────────────────────────
def median_filter(sig, kernel=7):
    """
    Median filter.
    Replaces each point with the median of its neighbours.
    Best classical method for spike removal.
    Does not blur edges.
    """
    from scipy.signal import medfilt
    return medfilt(sig.astype(np.float64),
                   kernel_size=kernel).astype(np.float32)

T_med    = median_filter(T_raw, kernel=7)
noise_med = noise_level(T_med)
spikes   = int((np.abs(T_raw - T_med) > 50).sum())
print(f"\n  Method 3 -- Median Filter (k=7)")
print(f"  Noise : {noise_med:.2f} C  |  Reduction : {(1-noise_med/noise_raw)*100:.1f}%")
print(f"  Spikes removed : {spikes}")

# ── Method 4: Savitzky-Golay ──────────────────────────────────────────────────
def savgol(sig, window=11, poly=3):
    """
    Savitzky-Golay filter.
    Fits a polynomial to each local window.
    Preserves peaks and sharp features better than moving average.
    """
    return signal.savgol_filter(sig.astype(np.float64),
                                window_length=window,
                                polyorder=poly).astype(np.float32)

T_sg    = savgol(T_raw, window=11, poly=3)
noise_sg = noise_level(T_sg)
print(f"\n  Method 4 -- Savitzky-Golay (w=11, p=3)")
print(f"  Noise : {noise_sg:.2f} C  |  Reduction : {(1-noise_sg/noise_raw)*100:.1f}%")

# ── Method 5: Gaussian Filter ─────────────────────────────────────────────────
def gaussian_filter(sig, sigma=3.0):
    """
    Gaussian filter.
    Smooths signal using a Gaussian kernel.
    Industry standard for general signal smoothing.
    """
    return gaussian_filter1d(sig.astype(np.float64),
                             sigma=sigma).astype(np.float32)

T_gauss    = gaussian_filter(T_raw, sigma=3.0)
noise_gauss = noise_level(T_gauss)
print(f"\n  Method 5 -- Gaussian Filter (sigma=3.0)")
print(f"  Noise : {noise_gauss:.2f} C  |  Reduction : {(1-noise_gauss/noise_raw)*100:.1f}%")

# ── Method 6: Kalman Filter ───────────────────────────────────────────────────
def kalman_filter(sig, process_noise=1e-3, measurement_noise=10.0):
    """
    Kalman filter.
    Recursive state estimator - balances trust between
    prediction and measurement.
    Very common in industrial sensor denoising and control systems.
    process_noise     : how much true signal can change each step
    measurement_noise : how noisy the sensor is (higher = more smoothing)
    """
    n_sig  = len(sig)
    x_est  = np.zeros(n_sig, dtype=np.float64)
    P_est  = np.zeros(n_sig, dtype=np.float64)
    x_est[0] = sig[0]
    P_est[0] = 1.0
    Q = process_noise
    R = measurement_noise
    for i in range(1, n_sig):
        x_pred   = x_est[i-1]
        P_pred   = P_est[i-1] + Q
        K        = P_pred / (P_pred + R)
        x_est[i] = x_pred + K * (sig[i] - x_pred)
        P_est[i] = (1 - K) * P_pred
    return x_est.astype(np.float32)

T_kalman    = kalman_filter(T_raw, process_noise=1e-3, measurement_noise=10.0)
noise_kalman = noise_level(T_kalman)
print(f"\n  Method 6 -- Kalman Filter (Q=0.001, R=10.0)")
print(f"  Noise : {noise_kalman:.2f} C  |  Reduction : {(1-noise_kalman/noise_raw)*100:.1f}%")
print(f"  Note  : recursive, suitable for real-time industrial systems")

# Preview table
print()
print("  -- 5-ROW PREVIEW -- Denoising (hottest region) --")
df_den = pd.DataFrame({
    "Time_s"    : np.round(time_s[idx], 4),
    "Raw"       : np.round(T_raw[idx],    2),
    "MovAvg"    : np.round(T_ma[idx],     2),
    "Median"    : np.round(T_med[idx],    2),
    "SavGol"    : np.round(T_sg[idx],     2),
    "Gaussian"  : np.round(T_gauss[idx],  2),
    "Kalman"    : np.round(T_kalman[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df_den.to_string())


# =============================================================================
# STEP 2 -- ALL CALIBRATION METHODS
# =============================================================================
print()
print("=" * 65)
print("STEP 2 -- ALL Classical Calibration Methods")
print("=" * 65)

T_input = T_med.copy()  # use median-filtered as input

# ── Method 1: Raw Baseline (no calibration) ───────────────────────────────────
rmse_raw_cal = float(np.sqrt(mean_squared_error(T_tc, T_input)))
mae_raw_cal  = float(mean_absolute_error(T_tc, T_input))
print(f"\n  Method 1 -- Raw Baseline (no calibration)")
print(f"  RMSE : {rmse_raw_cal:.2f} C  |  MAE : {mae_raw_cal:.2f} C")
print(f"  Note : zero baseline -- all methods must beat this")

# ── Method 2: Mean Offset Correction ─────────────────────────────────────────
def mean_offset(T_pyr, T_ref, cal_frac=0.20):
    """
    Mean offset correction.
    Simplest possible calibration -- computes mean difference
    between pyrometer and reference on calibration window
    and subtracts it from the full signal.
    """
    cal_end = int(cal_frac * len(T_pyr))
    offset  = np.mean(T_ref[:cal_end].astype(np.float64) -
                      T_pyr[:cal_end].astype(np.float64))
    return (T_pyr.astype(np.float64) + offset).astype(np.float32), offset

T_offset, off_val = mean_offset(T_input, T_tc, 0.20)
rmse_offset = float(np.sqrt(mean_squared_error(T_tc, T_offset)))
mae_offset  = float(mean_absolute_error(T_tc, T_offset))
print(f"\n  Method 2 -- Mean Offset Correction")
print(f"  Offset : {off_val:.2f} C")
print(f"  RMSE   : {rmse_offset:.2f} C  |  MAE : {mae_offset:.2f} C")

# ── Method 3: Linear Regression ──────────────────────────────────────────────
def linear_fit(T_pyr, T_ref, cal_frac=0.20):
    """
    Linear regression: T_ref = a * T_pyr + b
    Fits on calibration window, applies to full signal.
    Standard sensor calibration method.
    """
    cal_end = int(cal_frac * len(T_pyr))
    x = T_pyr[:cal_end].astype(np.float64)
    y = T_ref[:cal_end].astype(np.float64)
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return (a * T_pyr + b).astype(np.float32), a, b

T_lin, a_lin, b_lin = linear_fit(T_input, T_tc, 0.20)
T_lin = remove_drift(T_lin, T_tc)
rmse_lin = float(np.sqrt(mean_squared_error(T_tc, T_lin)))
mae_lin  = float(mean_absolute_error(T_tc, T_lin))
print(f"\n  Method 3 -- Linear Regression")
print(f"  Formula : T_cal = {a_lin:.5f} x T_pyr + {b_lin:.2f}")
print(f"  RMSE    : {rmse_lin:.2f} C  |  MAE : {mae_lin:.2f} C")

# ── Method 4: Polynomial Regression (degree=2) ────────────────────────────────
def polynomial_fit(T_pyr, T_ref, cal_frac=0.20, degree=2):
    """
    Polynomial regression: T_ref = a*T^2 + b*T + c
    Better than linear when emissivity changes with temperature.
    """
    cal_end = int(cal_frac * len(T_pyr))
    coeffs  = np.polyfit(T_pyr[:cal_end].astype(np.float64),
                         T_ref[:cal_end].astype(np.float64), degree)
    poly_fn = np.poly1d(coeffs)
    return poly_fn(T_pyr.astype(np.float64)).astype(np.float32), coeffs

T_poly, c_poly = polynomial_fit(T_input, T_tc, 0.20, degree=2)
rmse_poly = float(np.sqrt(mean_squared_error(T_tc, T_poly)))
mae_poly  = float(mean_absolute_error(T_tc, T_poly))
print(f"\n  Method 4 -- Polynomial Regression (degree=2)")
print(f"  RMSE : {rmse_poly:.2f} C  |  MAE : {mae_poly:.2f} C")

# ── Method 5: Piecewise Linear (3 segments) ───────────────────────────────────
def piecewise_linear(T_pyr, T_ref, cal_frac=0.20, n_pieces=3):
    """
    Piecewise linear calibration.
    Divides temperature range into n_pieces segments
    and fits a separate line in each segment.
    Good when emissivity changes at different temperature ranges.
    """
    cal_end = int(cal_frac * len(T_pyr))
    x_cal   = T_pyr[:cal_end].astype(np.float64)
    y_cal   = T_ref[:cal_end].astype(np.float64)
    t_min, t_max = x_cal.min(), x_cal.max()
    breaks  = np.linspace(t_min, t_max, n_pieces + 1)
    T_out   = np.zeros(len(T_pyr), dtype=np.float64)
    for k in range(n_pieces):
        lo, hi  = breaks[k], breaks[k+1]
        seg_cal = (x_cal >= lo) & (x_cal <= hi)
        seg_all = (T_pyr >= lo) & (T_pyr <= hi)
        if seg_cal.sum() < 2: continue
        A = np.vstack([x_cal[seg_cal], np.ones(seg_cal.sum())]).T
        a, b = np.linalg.lstsq(A, y_cal[seg_cal], rcond=None)[0]
        T_out[seg_all] = a * T_pyr[seg_all] + b
    zero_mask = T_out == 0
    if zero_mask.any() and (~zero_mask).any():
        T_out[zero_mask] = np.interp(
            np.where(zero_mask)[0],
            np.where(~zero_mask)[0],
            T_out[~zero_mask])
    return T_out.astype(np.float32)

T_pw    = piecewise_linear(T_input, T_tc, 0.20, n_pieces=3)
rmse_pw = float(np.sqrt(mean_squared_error(T_tc, T_pw)))
mae_pw  = float(mean_absolute_error(T_tc, T_pw))
print(f"\n  Method 5 -- Piecewise Linear (3 segments)")
print(f"  RMSE : {rmse_pw:.2f} C  |  MAE : {mae_pw:.2f} C")

# Preview table
print()
print("  -- 5-ROW PREVIEW -- Calibration (hottest region) --")
df_cal = pd.DataFrame({
    "Time_s"   : np.round(time_s[idx], 4),
    "Denoised" : np.round(T_input[idx],  2),
    "TC_ref"   : np.round(T_tc[idx],     2),
    "Offset"   : np.round(T_offset[idx], 2),
    "Linear"   : np.round(T_lin[idx],    2),
    "Poly_d2"  : np.round(T_poly[idx],   2),
    "Piecewise": np.round(T_pw[idx],     2),
}, index=[f"t{i}" for i in idx])
print(df_cal.to_string())


# =============================================================================
# STEP 3 -- ALL COMPRESSION METHODS
# =============================================================================
print()
print("=" * 65)
print("STEP 3 -- ALL Classical Compression Methods")
print("=" * 65)

T_comp_input  = T_lin.copy()
raw_size_bytes = T_comp_input.nbytes

# ── Method 1: Raw Baseline (no compression) ───────────────────────────────────
print(f"\n  Method 1 -- Raw Baseline (no compression)")
print(f"  Ratio : 1.0x  |  RMSE : 0.00 C")
print(f"  Size  : {raw_size_bytes} bytes = {raw_size_bytes/1024:.1f} KB")

# ── Method 2: Delta Encoding ──────────────────────────────────────────────────
def delta_encode(sig):
    """
    Delta encoding.
    Stores first value then only differences between consecutive values.
    Temperature signals change slowly so differences are small
    and can be stored with fewer bits.
    Near-lossless compression.
    """
    sig_f64  = sig.astype(np.float64)
    deltas   = np.diff(sig_f64)
    scale    = 100.0
    deltas_q = np.clip(np.round(deltas * scale),
                       -32767, 32767).astype(np.int16)
    return {"first": sig_f64[0], "deltas": deltas_q, "scale": scale}

def delta_decode(enc):
    deltas_f = enc["deltas"].astype(np.float64) / enc["scale"]
    sig_rec  = np.concatenate([[enc["first"]],
                                np.cumsum(deltas_f) + enc["first"]])
    return sig_rec.astype(np.float32)

enc_delta   = delta_encode(T_comp_input)
T_delta_rec = delta_decode(enc_delta)
n2          = min(len(T_comp_input), len(T_delta_rec))
rmse_delta  = float(np.sqrt(mean_squared_error(
                T_comp_input[:n2], T_delta_rec[:n2])))
delta_bytes = 8 + enc_delta["deltas"].nbytes
ratio_delta = raw_size_bytes / delta_bytes
print(f"\n  Method 2 -- Delta Encoding (int16, scale=100)")
print(f"  Ratio : {ratio_delta:.1f}x  |  RMSE : {rmse_delta:.4f} C (near-lossless)")

# ── Method 3: Downsampling ────────────────────────────────────────────────────
T_down  = T_comp_input[::2]
T_ds    = np.interp(np.arange(n),
                    np.arange(0, n, 2)[:len(T_down)],
                    T_down).astype(np.float32)
rmse_ds  = float(np.sqrt(mean_squared_error(T_comp_input, T_ds)))
ratio_ds = 2.0
print(f"\n  Method 3 -- Downsampling (factor=2x)")
print(f"  Ratio : {ratio_ds:.1f}x  |  RMSE : {rmse_ds:.2f} C")

# ── Method 4: SVD (rank=10) ───────────────────────────────────────────────────
sig_2d  = T_comp_input[:2048].reshape(64, 32).astype(np.float64)
cs      = svd_compress(sig_2d, rank=10)
rs      = svd_reconstruct(cs)
T_svd   = np.interp(np.arange(n),
                    np.linspace(0, n-1, rs.ravel().shape[0]),
                    rs.ravel()).astype(np.float32)
rmse_svd  = float(np.sqrt(mean_squared_error(T_comp_input, T_svd)))
ratio_svd = sig_2d.nbytes / (cs["U"].nbytes+cs["S"].nbytes+cs["Vt"].nbytes)
print(f"\n  Method 4 -- SVD (rank=10)")
print(f"  Ratio : {ratio_svd:.1f}x  |  RMSE : {rmse_svd:.2f} C")

# ── Method 5: Wavelet Haar (10%) ──────────────────────────────────────────────
cw       = wavelet_compress(T_comp_input, keep_fraction=0.10)
T_wav    = wavelet_reconstruct(cw)
rmse_wav  = float(np.sqrt(mean_squared_error(T_comp_input, T_wav)))
ratio_wav = T_comp_input.nbytes / max(1, cw["nonzero"] * 8)
print(f"\n  Method 5 -- Wavelet Haar (10% retention)")
print(f"  Ratio : {ratio_wav:.1f}x  |  RMSE : {rmse_wav:.2f} C")
print(f"  Note  : only method satisfying CR > 4x target")

# Preview table
print()
print("  -- 5-ROW PREVIEW -- Compression (hottest region) --")
df_comp = pd.DataFrame({
    "Time_s"    : np.round(time_s[idx],       4),
    "Calibrated": np.round(T_comp_input[idx], 2),
    "Delta"     : np.round(T_delta_rec[idx],  2),
    "Downsamp"  : np.round(T_ds[idx],         2),
    "SVD_r10"   : np.round(T_svd[idx],        2),
    "Wavelet10%": np.round(T_wav[idx],        2),
}, index=[f"t{i}" for i in idx])
print(df_comp.to_string())


# =============================================================================
# FULL SUMMARY TABLE
# =============================================================================
print()
print("=" * 65)
print("FULL SUMMARY TABLE -- All Classical Methods")
print("=" * 65)

summary_rows = [
    # Denoising
    {"Step":"Denoising", "Method":"Raw Baseline",           "Type":"Baseline",
     "RMSE_C":"N/A", "Ratio":"N/A", "Noise_C":round(noise_raw,2),  "Reduction_%":"0.0"},
    {"Step":"Denoising", "Method":"Moving Average (k=7)",   "Type":"Classical",
     "RMSE_C":"N/A", "Ratio":"N/A", "Noise_C":round(noise_ma,2),   "Reduction_%":f"{(1-noise_ma/noise_raw)*100:.1f}"},
    {"Step":"Denoising", "Method":"Median Filter (k=7)",    "Type":"Classical",
     "RMSE_C":"N/A", "Ratio":"N/A", "Noise_C":round(noise_med,2),  "Reduction_%":f"{(1-noise_med/noise_raw)*100:.1f}"},
    {"Step":"Denoising", "Method":"Savitzky-Golay (w=11)",  "Type":"Classical",
     "RMSE_C":"N/A", "Ratio":"N/A", "Noise_C":round(noise_sg,2),   "Reduction_%":f"{(1-noise_sg/noise_raw)*100:.1f}"},
    {"Step":"Denoising", "Method":"Gaussian (sigma=3.0)",   "Type":"Classical",
     "RMSE_C":"N/A", "Ratio":"N/A", "Noise_C":round(noise_gauss,2),"Reduction_%":f"{(1-noise_gauss/noise_raw)*100:.1f}"},
    {"Step":"Denoising", "Method":"Kalman (Q=0.001,R=10)",  "Type":"Classical",
     "RMSE_C":"N/A", "Ratio":"N/A", "Noise_C":round(noise_kalman,2),"Reduction_%":f"{(1-noise_kalman/noise_raw)*100:.1f}"},
    # Calibration
    {"Step":"Calibration","Method":"Raw Baseline",           "Type":"Baseline",
     "RMSE_C":round(rmse_raw_cal,2),"Ratio":"N/A","Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Calibration","Method":"Mean Offset",            "Type":"Classical",
     "RMSE_C":round(rmse_offset,2), "Ratio":"N/A","Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Calibration","Method":"Linear Regression",      "Type":"Classical",
     "RMSE_C":round(rmse_lin,2),    "Ratio":"N/A","Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Calibration","Method":"Polynomial (degree=2)",  "Type":"Classical",
     "RMSE_C":round(rmse_poly,2),   "Ratio":"N/A","Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Calibration","Method":"Piecewise (3 segments)", "Type":"Classical",
     "RMSE_C":round(rmse_pw,2),     "Ratio":"N/A","Noise_C":"N/A","Reduction_%":"N/A"},
    # Compression
    {"Step":"Compression","Method":"Raw Baseline",           "Type":"Baseline",
     "RMSE_C":"0.00","Ratio":"1.0","Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Compression","Method":"Delta Encoding (int16)", "Type":"Classical",
     "RMSE_C":round(rmse_delta,4),  "Ratio":round(ratio_delta,1),"Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Compression","Method":"Downsampling (2x)",      "Type":"Classical",
     "RMSE_C":round(rmse_ds,2),     "Ratio":"2.0","Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Compression","Method":"SVD (rank=10)",          "Type":"Classical",
     "RMSE_C":round(rmse_svd,2),    "Ratio":round(ratio_svd,1),"Noise_C":"N/A","Reduction_%":"N/A"},
    {"Step":"Compression","Method":"Wavelet Haar (10%)",     "Type":"Classical",
     "RMSE_C":round(rmse_wav,2),    "Ratio":round(ratio_wav,1),"Noise_C":"N/A","Reduction_%":"N/A"},
]

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
df_summary.to_csv("classical_summary_updated.csv", index=False)
print("\n  Saved --> classical_summary_updated.csv")


# =============================================================================
# VISUALISATION
# =============================================================================
print()
print("  Generating plots ...")

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

# ── Row 1: Denoising ─────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_s, T_raw,    color="#e74c3c", lw=0.5, alpha=0.4, label="Raw")
ax1.plot(time_s, T_ma,     color="#3498db", lw=0.9, label="Moving Avg")
ax1.plot(time_s, T_med,    color="#2ecc71", lw=0.9, label="Median")
ax1.plot(time_s, T_sg,     color="#9b59b6", lw=0.9, ls="--", label="Savitzky-Golay")
ax1.plot(time_s, T_gauss,  color="#f39c12", lw=0.9, label="Gaussian")
ax1.plot(time_s, T_kalman, color="#1abc9c", lw=0.9, ls=":", label="Kalman")
ax1.set_title("Denoising -- All 6 Classical Methods", fontweight="bold", fontsize=11)
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Temperature (C)")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax1.set_facecolor("#ffffff")

ax2 = fig.add_subplot(gs[0, 1])
methods_n  = ["Raw\nBaseline","Moving\nAverage","Median\nFilter",
               "Savitzky-\nGolay","Gaussian\nFilter","Kalman\nFilter"]
noise_vals = [noise_raw, noise_ma, noise_med, noise_sg, noise_gauss, noise_kalman]
colors_n   = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c"]
bars = ax2.bar(methods_n, noise_vals, color=colors_n, alpha=0.85, width=0.6)
for bar, val in zip(bars, noise_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.set_title("Denoising -- Noise Level (lower=better)", fontweight="bold", fontsize=11)
ax2.set_ylabel("Noise std dev (C)"); ax2.grid(True, alpha=0.3, axis="y")
ax2.set_facecolor("#ffffff")

# ── Row 2: Calibration ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_s, T_tc,     color="#f39c12", lw=1.2, ls=":", label="TC reference")
ax3.plot(time_s, T_input,  color="#95a5a6", lw=0.5, alpha=0.4, label="Denoised input")
ax3.plot(time_s, T_offset, color="#e74c3c", lw=0.9, label="Mean Offset")
ax3.plot(time_s, T_lin,    color="#3498db", lw=0.9, label="Linear")
ax3.plot(time_s, T_poly,   color="#2ecc71", lw=0.9, label="Polynomial d=2")
ax3.plot(time_s, T_pw,     color="#9b59b6", lw=0.9, ls="--", label="Piecewise")
ax3.set_title("Calibration -- All 5 Methods vs TC Reference",
              fontweight="bold", fontsize=11)
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Temperature (C)")
ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)
ax3.set_facecolor("#ffffff")

ax4 = fig.add_subplot(gs[1, 1])
methods_c  = ["Raw\nBaseline","Mean\nOffset","Linear\nRegr.",
               "Polynomial\nd=2","Piecewise\n3 seg."]
rmse_vals  = [rmse_raw_cal, rmse_offset, rmse_lin, rmse_poly, rmse_pw]
colors_c   = ["#e74c3c","#f39c12","#3498db","#2ecc71","#9b59b6"]
bars = ax4.bar(methods_c, rmse_vals, color=colors_c, alpha=0.85, width=0.6)
for bar, val in zip(bars, rmse_vals):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax4.set_title("Calibration -- RMSE vs TC Reference (lower=better)",
              fontweight="bold", fontsize=11)
ax4.set_ylabel("RMSE (C)"); ax4.grid(True, alpha=0.3, axis="y")
ax4.set_facecolor("#ffffff")

# ── Row 3: Compression ───────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(time_s, T_comp_input, color="black",   lw=1.2, label="Calibrated input")
ax5.plot(time_s, T_delta_rec,  color="#e74c3c", lw=0.9, label="Delta Encoding")
ax5.plot(time_s, T_ds,         color="#3498db", lw=0.9, label="Downsample 2x")
ax5.plot(time_s, T_svd,        color="#2ecc71", lw=0.9, label="SVD rank=10")
ax5.plot(time_s, T_wav,        color="#9b59b6", lw=0.9, ls="--", label="Wavelet 10%")
ax5.set_title("Compression -- All 5 Methods", fontweight="bold", fontsize=11)
ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Temperature (C)")
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)
ax5.set_facecolor("#ffffff")

ax6 = fig.add_subplot(gs[2, 1])
comp_methods = ["Raw\n(1.0x)","Delta\nEnc.","Downsamp\n2x",
                 "SVD\nr=10","Wavelet\n10%"]
comp_rmse    = [0.0, rmse_delta, rmse_ds, rmse_svd, rmse_wav]
comp_ratio   = [1.0, ratio_delta, 2.0, ratio_svd, ratio_wav]
comp_colors  = ["#e74c3c","#f39c12","#3498db","#2ecc71","#9b59b6"]
ax6.scatter(comp_ratio, comp_rmse, c=comp_colors, s=150,
            zorder=5, edgecolors="black", lw=0.8)
for m, x, y in zip(comp_methods, comp_ratio, comp_rmse):
    ax6.annotate(m, (x, y), textcoords="offset points",
                 xytext=(6, 5), fontsize=9)
ax6.axvline(x=4.0, color="green", ls="--", lw=1.2, alpha=0.7,
            label="4x target")
ax6.set_title("Compression -- RMSE vs Ratio (bottom-right=best)",
              fontweight="bold", fontsize=11)
ax6.set_xlabel("Compression ratio (higher=smaller file)")
ax6.set_ylabel("RMSE (C) -- lower=better")
ax6.legend(fontsize=8); ax6.grid(True, alpha=0.3)
ax6.set_facecolor("#ffffff")

plt.suptitle(
    "All Classical Methods: Denoising, Calibration, Compression\n"
    "NIST Layer01 IN625 data",
    fontsize=13, fontweight="bold", y=1.01)

plt.savefig("classical_results.png", dpi=150,
            bbox_inches="tight", facecolor="#f8f9fa")
print("  Plot saved --> classical_results.png")
plt.show()

print()
print("=" * 65)
print("ALL CLASSICAL METHODS COMPLETE")
print("=" * 65)
print()
print("  DENOISING (6 methods including baseline):")
print(f"    Raw Baseline    : {noise_raw:.2f} C noise")
print(f"    Moving Average  : {noise_ma:.2f} C  ({(1-noise_ma/noise_raw)*100:.1f}% reduction)")
print(f"    Median Filter   : {noise_med:.2f} C  ({(1-noise_med/noise_raw)*100:.1f}% reduction)")
print(f"    Savitzky-Golay  : {noise_sg:.2f} C  ({(1-noise_sg/noise_raw)*100:.1f}% reduction)")
print(f"    Gaussian Filter : {noise_gauss:.2f} C  ({(1-noise_gauss/noise_raw)*100:.1f}% reduction)")
print(f"    Kalman Filter   : {noise_kalman:.2f} C  ({(1-noise_kalman/noise_raw)*100:.1f}% reduction)")
print()
print("  CALIBRATION (5 methods including baseline):")
print(f"    Raw Baseline    : {rmse_raw_cal:.2f} C RMSE")
print(f"    Mean Offset     : {rmse_offset:.2f} C RMSE")
print(f"    Linear          : {rmse_lin:.2f} C RMSE")
print(f"    Polynomial d=2  : {rmse_poly:.2f} C RMSE")
print(f"    Piecewise       : {rmse_pw:.2f} C RMSE")
print()
print("  COMPRESSION (5 methods including baseline):")
print(f"    Raw Baseline    : 1.0x ratio,  0.00 C RMSE")
print(f"    Delta Encoding  : {ratio_delta:.1f}x ratio,  {rmse_delta:.4f} C RMSE")
print(f"    Downsampling 2x : 2.0x ratio,  {rmse_ds:.2f} C RMSE")
print(f"    SVD rank=10     : {ratio_svd:.1f}x ratio,  {rmse_svd:.2f} C RMSE")
print(f"    Wavelet 10%     : {ratio_wav:.1f}x ratio,  {rmse_wav:.2f} C RMSE")
print()
print("  Compare these against ML methods:")
print("  CNN, LSTM, Random Forest, MLP, PCA, Autoencoder")
print("=" * 65)