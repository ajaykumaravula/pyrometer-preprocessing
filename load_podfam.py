"""
=============================================================================
load_podfam.py  --  Complete PODFAM Pipeline: All Classical Methods
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)
Dataset: AP&T PODFAM .pcd files (Aconity STUDIO laser system)

PIPELINE STAGES:

    Stage 1 -- Denoising (ATP-1) -- Classical methods only:
        1. Raw Baseline
        2. Moving Average  (k=7)
        3. Median Filter   (k=7)  <- primary method
        4. Savitzky-Golay  (w=11, p=3)
        5. Gaussian Filter (sigma=3.0)
        6. Kalman Filter   (Q=0.001, R=10.0)

    Stage 2 -- Calibration (ATP-2):
        Two-colour formula (provided by Karthikeyan, AP&T)
        T[K] = 2044.7 / (ln(S1_corr/S0_corr) + 0.83)
        Note: No thermocouple available in PODFAM files.
              ML calibration methods cannot be applied.

    Stage 3 -- Compression (ATP-3) -- Classical methods only:
        1. Raw Baseline
        2. Delta Encoding (int16)
        3. Downsampling   (2x)
        4. SVD            (rank=10)
        5. Wavelet Haar   (10%)   <- primary method

NOTE: ML methods (CNN, LSTM, RF, MLP, PCA, AE etc.) are NOT applied
      to PODFAM because:
      - No thermocouple reference for calibration training
      - ML denoisers trained on NIST 1D signal cannot directly apply
        to PODFAM spatial scan format without retraining

HOW TO RUN:
    python load_podfam.py

OUTPUT:
    podfam_result.png
    podfam_summary.csv
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(0, ".")

from compress import wavelet_compress, wavelet_reconstruct, \
                     svd_compress, svd_reconstruct

# =============================================================================
# CONFIGURATION
# =============================================================================
PCD_FILE = r"C:\Users\sravy\OneDrive\Desktop\Thesis\podfam_data\000.280.pcd"

# Two-colour calibration constants (provided by Karthikeyan, AP&T)
B1           = 2044.7
B2           = 0.83
BG_S0        = 792.45
BG_S1        = 798.45
TOP_N        = 100      # top N hottest pixels for melt pool temperature

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def noise_level(sig):
    """Noise standard deviation using 20-point moving average baseline."""
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

def moving_average(sig, k=7):
    """Moving average filter."""
    return np.convolve(sig, np.ones(k)/k, mode="same").astype(np.float32)

def median_filt(sig, k=7):
    """Median filter -- primary denoising method."""
    return medfilt(sig.astype(np.float64),
                   kernel_size=k).astype(np.float32)

def savgol(sig, w=11, p=3):
    """Savitzky-Golay filter."""
    return savgol_filter(sig.astype(np.float64),
                         window_length=w,
                         polyorder=p).astype(np.float32)

def gauss_filt(sig, sigma=3.0):
    """Gaussian filter."""
    return gaussian_filter1d(sig.astype(np.float64),
                             sigma=sigma).astype(np.float32)

def kalman_filt(sig, Q=1e-3, R=10.0):
    """Kalman filter."""
    x = float(sig[0]); P = 1.0
    out = np.zeros(len(sig), dtype=np.float32)
    for i, z in enumerate(sig):
        P_p = P + Q
        K   = P_p / (P_p + R)
        x   = x + K * (float(z) - x)
        P   = (1 - K) * P_p
        out[i] = x
    return out

def delta_encode(sig):
    """Delta encoding -- stores differences between consecutive values."""
    d   = np.diff(sig.astype(np.float64))
    dq  = np.clip(np.round(d * 100), -32767, 32767).astype(np.int16)
    return {"first": float(sig[0]), "deltas": dq, "scale": 100.0}

def delta_decode(enc):
    """Reconstruct signal from delta encoding."""
    d   = enc["deltas"].astype(np.float64) / enc["scale"]
    return np.concatenate([[enc["first"]],
                            np.cumsum(d) + enc["first"]]).astype(np.float32)

def two_colour_temp(s0_corr, s1_corr):
    """
    Two-colour pyrometer calibration formula (Karthikeyan, AP&T).
    T[K] = B1 / (ln(S1_corr/S0_corr) + B2)
    Returns temperature in degrees Celsius.
    """
    ratio = s1_corr / np.maximum(s0_corr, 1e-6)
    ratio = np.clip(ratio, 1e-6, None)
    T_K   = B1 / (np.log(ratio) + B2)
    return T_K - 273.15

# =============================================================================
# LOAD PCD FILE
# =============================================================================
print("=" * 65)
print("load_podfam.py -- Complete PODFAM Classical Pipeline")
print("=" * 65)
print(f"\n  Loading: {PCD_FILE}")

rows = []
with open(PCD_FILE, 'r') as f:
    started = False
    for line in f:
        if 'DATA ascii' in line:
            started = True; continue
        if started:
            vals = line.strip().split()
            if len(vals) == 10:
                rows.append([int(v) for v in vals])

df = pd.DataFrame(rows, columns=[
    't','x','y','z',
    'sensor0','sensor1',
    'sensor2','sensor3',
    'state0','state1'
])
df['time_s'] = (df['t'] - df['t'].min()) / 1e6

print(f"  Points loaded : {len(df):,}")
print(f"  Duration      : {df['time_s'].max():.2f} s")
print(f"  Sensor0 range : {df['sensor0'].min()} -- {df['sensor0'].max()}")
print(f"  Sensor1 range : {df['sensor1'].min()} -- {df['sensor1'].max()}")

# Raw signals
S0_raw  = df['sensor0'].values.astype(np.float32)
S1_raw  = df['sensor1'].values.astype(np.float32)
time_s  = df['time_s'].values
n       = len(S0_raw)
raw_bytes = S0_raw.nbytes

# Background correction
S0_corr = S0_raw - BG_S0
S1_corr = S1_raw - BG_S1

# 5-row preview
print("\n  5-ROW PREVIEW -- Raw data:")
print(df[['time_s','sensor0','sensor1']].head(5).to_string())

# =============================================================================
# STAGE 1 -- ALL CLASSICAL DENOISING METHODS (ATP-1)
# =============================================================================
print()
print("=" * 65)
print("STAGE 1 -- Denoising (ATP-1) -- All Classical Methods")
print("=" * 65)

noise_raw = noise_level(S0_raw)
print(f"\n  Raw Baseline   noise={noise_raw:.2f}  reduction=0.0%")

T_ma     = moving_average(S0_raw, 7)
T_med    = median_filt(S0_raw, 7)
T_sg     = savgol(S0_raw, 11, 3)
T_gauss  = gauss_filt(S0_raw, 3.0)
T_kalman = kalman_filt(S0_raw, 1e-3, 10.0)

den_results = []
for name, sig in [
    ("Moving Average (k=7)",   T_ma),
    ("Median Filter  (k=7)",   T_med),
    ("Savitzky-Golay (w=11)",  T_sg),
    ("Gaussian (sigma=3.0)",   T_gauss),
    ("Kalman  (Q=0.001,R=10)", T_kalman),
]:
    nl  = noise_level(sig)
    red = (1 - nl/noise_raw) * 100
    spk = int((np.abs(S0_raw - sig) > 5).sum())
    print(f"  {name:<28} noise={nl:.2f}  reduction={red:.1f}%  spikes={spk}")
    den_results.append({
        "Stage":"Denoise","Method":name,
        "Noise":round(nl,2),"Reduction_%":round(red,1),
        "Spikes_removed":spk
    })

# Use Median as primary denoised signal
T_den = T_med.copy()
print(f"\n  Primary denoised signal : Median Filter")

# =============================================================================
# STAGE 2 -- CALIBRATION (ATP-2) -- Two-colour formula only
# =============================================================================
print()
print("=" * 65)
print("STAGE 2 -- Calibration (ATP-2) -- Two-Colour Formula")
print("=" * 65)
print("  Note: No thermocouple in PODFAM files.")
print("        Using two-colour formula (Karthikeyan, AP&T).")

# Apply two-colour formula to top N hottest pixels
combined = S0_corr + S1_corr
top_idx  = np.argsort(combined)[-TOP_N:]
T_top    = two_colour_temp(S0_corr[top_idx], S1_corr[top_idx])
T_top    = T_top[np.isfinite(T_top)]

T_median_pool = float(np.median(T_top))
T_max_pool    = float(np.max(T_top))
T_min_pool    = float(np.min(T_top))

print(f"\n  Top {TOP_N} hottest pixels:")
print(f"  Melt pool T median : {T_median_pool:.1f} C")
print(f"  Melt pool T max    : {T_max_pool:.1f} C")
print(f"  Melt pool T min    : {T_min_pool:.1f} C")
print(f"  Note: 1500-1900 C expected for laser powder bed fusion")

# Also compute pixel-wise calibrated temperature for full signal
T_cal_full = two_colour_temp(
    np.maximum(S0_corr, 1e-6),
    np.maximum(S1_corr, 1e-6)
)
T_cal_full = np.clip(T_cal_full, -300, 3000).astype(np.float32)

# =============================================================================
# STAGE 3 -- ALL CLASSICAL COMPRESSION METHODS (ATP-3)
# =============================================================================
print()
print("=" * 65)
print("STAGE 3 -- Compression (ATP-3) -- All Classical Methods")
print("=" * 65)

# Use denoised sensor0 as compression input
T_comp = T_den.copy()

# Raw baseline
print(f"\n  Raw Baseline   ratio=1.0x  RMSE=0.00")

comp_results = []

# Delta Encoding
enc_delta   = delta_encode(T_comp)
T_delta     = delta_decode(enc_delta)
n2          = min(len(T_comp), len(T_delta))
rmse_delta  = float(np.sqrt(mean_squared_error(T_comp[:n2], T_delta[:n2])))
delta_bytes = 8 + enc_delta["deltas"].nbytes
ratio_delta = raw_bytes / delta_bytes
print(f"  Delta Encoding  ratio={ratio_delta:.1f}x  RMSE={rmse_delta:.4f}  (near-lossless)")
comp_results.append({"Method":"Delta Encoding","Ratio":round(ratio_delta,1),
                     "RMSE":round(rmse_delta,4)})

# Downsampling 2x
T_ds    = np.interp(np.arange(n),
                    np.arange(0, n, 2)[:len(T_comp[::2])],
                    T_comp[::2]).astype(np.float32)
rmse_ds = float(np.sqrt(mean_squared_error(T_comp, T_ds)))
print(f"  Downsampling 2x ratio=2.0x  RMSE={rmse_ds:.2f}")
comp_results.append({"Method":"Downsampling 2x","Ratio":2.0,"RMSE":round(rmse_ds,2)})

# SVD rank=10
chunk   = min(2048, len(T_comp))
s2d     = T_comp[:chunk].reshape(-1, 32).astype(np.float64) \
          if chunk >= 32 else None
if s2d is not None:
    cs      = svd_compress(s2d, rank=10)
    rs      = svd_reconstruct(cs)
    T_svd   = np.interp(np.arange(n),
                        np.linspace(0, n-1, rs.ravel().shape[0]),
                        rs.ravel()).astype(np.float32)
    rmse_svd  = float(np.sqrt(mean_squared_error(T_comp, T_svd)))
    ratio_svd = s2d.nbytes / (cs["U"].nbytes+cs["S"].nbytes+cs["Vt"].nbytes)
    print(f"  SVD rank=10     ratio={ratio_svd:.1f}x  RMSE={rmse_svd:.2f}")
    comp_results.append({"Method":"SVD rank=10","Ratio":round(ratio_svd,1),
                         "RMSE":round(rmse_svd,2)})

# Wavelet Haar 10% -- PRIMARY
cw        = wavelet_compress(T_comp, keep_fraction=0.10)
T_wav     = wavelet_reconstruct(cw).astype(np.float32)
n3        = min(len(T_comp), len(T_wav))
rmse_wav  = float(np.sqrt(mean_squared_error(T_comp[:n3], T_wav[:n3])))
ratio_wav = raw_bytes / max(1, cw["nonzero"] * 8)
print(f"  Wavelet 10%     ratio={ratio_wav:.1f}x  RMSE={rmse_wav:.2f}  *** PRIMARY ***")
comp_results.append({"Method":"Wavelet Haar 10%","Ratio":round(ratio_wav,1),
                     "RMSE":round(rmse_wav,2)})

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\n  File            : {PCD_FILE.split(chr(92))[-1]}")
print(f"  Points          : {n:,}")
print(f"  Duration        : {df['time_s'].max():.2f} s")
print()
print("  DENOISING:")
for r in den_results:
    print(f"    {r['Method']:<28} {r['Reduction_%']:.1f}% reduction")
print()
print("  CALIBRATION (Two-colour formula):")
print(f"    Melt pool temperature : {T_median_pool:.1f} C (median of top {TOP_N} pixels)")
print(f"    Range                 : {T_min_pool:.1f} -- {T_max_pool:.1f} C")
print()
print("  COMPRESSION:")
for r in comp_results:
    print(f"    {r['Method']:<20} ratio={r['Ratio']}x  RMSE={r['RMSE']}")

# Save summary CSV
df_summary = pd.DataFrame(den_results + comp_results)
df_summary.to_csv("podfam_summary.csv", index=False)
print("\n  Saved --> podfam_summary.csv")

# =============================================================================
# VISUALISATION
# =============================================================================
print("\n  Generating plots ...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

fname = PCD_FILE.split("\\")[-1]

# ── Row 1: All denoising methods ─────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :2])
ax.plot(time_s, S0_raw,   color="lightcoral", lw=0.4, alpha=0.5, label="Raw")
ax.plot(time_s, T_ma,     color="#3498db",    lw=0.8, label="Moving Avg")
ax.plot(time_s, T_med,    color="#2ecc71",    lw=0.9, label="Median")
ax.plot(time_s, T_sg,     color="#9b59b6",    lw=0.8, ls="--", label="SavGol")
ax.plot(time_s, T_gauss,  color="#f39c12",    lw=0.8, label="Gaussian")
ax.plot(time_s, T_kalman, color="#1abc9c",    lw=0.8, ls=":", label="Kalman")
ax.set_title(f"Denoising -- All 5 Classical Methods\n{fname}", fontweight="bold")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Sensor counts")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_facecolor("#ffffff")

# ── Noise level bar chart ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
methods_n = ["Raw","MovAvg","Median","SavGol","Gaussian","Kalman"]
noise_v   = [noise_raw,
             noise_level(T_ma), noise_level(T_med),
             noise_level(T_sg), noise_level(T_gauss), noise_level(T_kalman)]
colors_n  = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c"]
bars = ax.bar(methods_n, noise_v, color=colors_n, alpha=0.85)
for bar, val in zip(bars, noise_v):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_title("Noise Level (lower=better)", fontweight="bold")
ax.set_ylabel("Noise std dev"); ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#ffffff")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

# ── Row 2: Calibration ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, :2])
# Show sensor counts and melt pool region
ax.plot(time_s, S0_corr, color="#e74c3c", lw=0.4, alpha=0.4, label="S0 corrected")
ax.plot(time_s, S1_corr, color="#3498db", lw=0.4, alpha=0.4, label="S1 corrected")
# Highlight top N pixels
ax.scatter(time_s[top_idx], S0_corr[top_idx],
           color="#f39c12", s=5, zorder=5, label=f"Top {TOP_N} hottest")
ax.set_title("Calibration -- Two-Colour Formula (Sensor0 + Sensor1)", fontweight="bold")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Corrected counts")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_facecolor("#ffffff")

ax = fig.add_subplot(gs[1, 2])
temps = [T_min_pool, T_median_pool, T_max_pool]
labels = ["Min", "Median\n(hotspot)", "Max"]
colors_t = ["#3498db", "#e74c3c", "#f39c12"]
bars = ax.bar(labels, temps, color=colors_t, alpha=0.85, width=0.5)
for bar, val in zip(bars, temps):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f"{val:.0f}°C", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_title("Melt Pool Temperature\n(Two-Colour Formula)", fontweight="bold")
ax.set_ylabel("Temperature (C)")
ax.set_ylim(0, T_max_pool * 1.2)
ax.axhline(y=1500, color="green", ls="--", lw=1.2, alpha=0.7, label="Expected min")
ax.axhline(y=2000, color="red",   ls="--", lw=1.2, alpha=0.7, label="Expected max")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#ffffff")

# ── Row 3: Compression ────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, :2])
ax.plot(time_s, T_comp,    color="black",   lw=1.0, label="Denoised (input)")
ax.plot(time_s, T_delta,   color="#e74c3c", lw=0.8, label=f"Delta {ratio_delta:.1f}x")
ax.plot(time_s, T_ds,      color="#3498db", lw=0.8, ls="--", label="Downsamp 2x")
if s2d is not None:
    ax.plot(time_s, T_svd, color="#9b59b6", lw=0.8, ls="-.", label=f"SVD {ratio_svd:.1f}x")
ax.plot(time_s[:n3], T_wav[:n3], color="#2ecc71", lw=0.9, ls=":",
        label=f"Wavelet {ratio_wav:.1f}x")
ax.set_title("Compression -- All 4 Classical Methods", fontweight="bold")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Sensor counts")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_facecolor("#ffffff")

ax = fig.add_subplot(gs[2, 2])
comp_names  = ["Delta\nEnc","Downsamp\n2x","SVD\nr=10","Wavelet\n10%"]
comp_ratios = [ratio_delta, 2.0,
               ratio_svd if s2d is not None else 0, ratio_wav]
comp_rmses  = [rmse_delta, rmse_ds,
               rmse_svd if s2d is not None else 0, rmse_wav]
colors_c    = ["#e74c3c","#3498db","#9b59b6","#2ecc71"]
ax.scatter(comp_ratios, comp_rmses, c=colors_c, s=150, zorder=5,
           edgecolors="black", lw=0.8)
for name, x, y in zip(comp_names, comp_ratios, comp_rmses):
    ax.annotate(name, (x, y), textcoords="offset points",
                xytext=(6, 5), fontsize=8)
ax.axvline(x=4.0, color="green", ls="--", lw=1.2, label="4x target")
ax.set_title("RMSE vs Ratio\n(bottom-right=best)", fontweight="bold")
ax.set_xlabel("Compression ratio"); ax.set_ylabel("RMSE")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_facecolor("#ffffff")

plt.suptitle(
    f"PODFAM Complete Classical Pipeline\nFile: {fname}",
    fontsize=13, fontweight="bold", y=1.01)

plt.savefig("podfam_result.png", dpi=150,
            bbox_inches="tight", facecolor="#f8f9fa")
print("  Plot saved --> podfam_result.png")
plt.show()

print()
print("=" * 65)
print("PODFAM PIPELINE COMPLETE")
print("=" * 65)
print(f"  Denoising  : 5 classical methods compared")
print(f"  Calibration: Two-colour formula")
print(f"               Melt pool T = {T_median_pool:.1f} C")
print(f"  Compression: 4 classical methods compared")
print(f"               Best: Wavelet {ratio_wav:.1f}x, RMSE={rmse_wav:.2f}")
print()
print("  NOTE: ML methods not applied to PODFAM because:")
print("  - No thermocouple reference for calibration training")
print("  - ML denoisers need retraining on PODFAM signal format")
print("=" * 65)
