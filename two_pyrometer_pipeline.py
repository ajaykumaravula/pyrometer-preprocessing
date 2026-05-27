"""
=============================================================================
Simulated 2-Pyrometer + Thermocouple Pre-Processing Pipeline
=============================================================================
Thesis: Automation of pyrometer data pre-processing
        (denoising, calibration, compression) — AP&T / Metal Forming

WHY SIMULATION?
  The real AP&T 2-pyrometer + thermocouple dataset is not yet available.
  This script builds a realistic simulation using the NIST Layer01.mat
  hot-spot temperature signal as the true underlying temperature, then
  adds physically realistic errors to create:

    • Pyrometer 1  — emissivity error + Gaussian noise + random spikes
    • Pyrometer 2  — different emissivity + different noise + offset drift
    • Thermocouple — clean reference with slight thermal lag

  When real AP&T data arrives, replace the "LOAD DATA" block only.
  Every downstream step (denoising, calibration, compression, plots)
  works unchanged.

HOW TO RUN:
    python two_pyrometer_pipeline.py

WHAT YOU SEE IN THE TERMINAL (5×5 preview at every stage):
    RAW          — what each sensor sees before processing
    DENOISED     — after spike removal + smoothing
    CALIBRATED   — after correcting emissivity using thermocouple
    FUSED        — single best-estimate temperature from both pyrometers
=============================================================================
"""

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.signal import medfilt

np.random.seed(42)   # reproducible results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD BASE SIGNAL FROM LAYER01 AND BUILD SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 0 — Building simulated 2-pyrometer + thermocouple dataset")
print("=" * 65)

# --- Load NIST Layer01 to extract the underlying hot-spot temperature --------
#mat   = sio.loadmat(r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat")
mat   = sio.loadmat(r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer03.mat")
# ↑ CHANGE THIS PATH if your folder is different. Use the same path that
#   worked in layer01_pipeline.py
L     = mat["Layer"][0, 0]
raw3d = L["RadiantTemp"].astype(np.float32)   # (126, 360, 2497)
sh_A  = float(L["SHvariable_A"].flat[0])      # 2.655
sh_B  = float(L["SHvariable_B"].flat[0])      # -800.7

# Max pixel per frame → best estimate of the melt-pool peak temperature
frame_max   = raw3d.max(axis=(0, 1))                       # (2497,)
T_true_K    = np.clip(sh_A * frame_max + sh_B, 273, 3000)  # Kelvin
T_true_C    = T_true_K - 273.15                             # Celsius

# Keep only frames that have a real signal (laser on)
signal_mask = T_true_C > 10
T_true_C    = T_true_C[signal_mask]
n           = len(T_true_C)
time_s      = np.linspace(0, n * 0.002, n)   # 500 Hz sampling → 2 ms per step

print(f"  Base signal       : {n} frames with laser signal")
print(f"  True temp range   : {T_true_C.min():.1f} °C — {T_true_C.max():.1f} °C")
print(f"  Simulated duration: {time_s[-1]:.2f} s")

# ── Pyrometer 1 — lower emissivity error, moderate noise ────────────────────
# Emissivity ε₁ = 0.85 (true = 1.0) → pyrometer reads high (Stefan-Boltzmann)
# Simplified: T_pyr = T_true × (1/ε)^0.25  (radiation pyrometer model)
eps1          = 0.85
T_pyr1_ideal  = T_true_C * (1.0 / eps1) ** 0.25
noise1        = np.random.normal(0, 12, n)         # ±12 °C Gaussian noise
spikes1       = np.zeros(n)
spike_idx1    = np.random.choice(n, size=int(n * 0.015), replace=False)
spikes1[spike_idx1] = np.random.uniform(200, 600, len(spike_idx1))
T_pyr1_raw    = T_pyr1_ideal + noise1 + spikes1    # what Pyrometer 1 records

# ── Pyrometer 2 — higher emissivity error, different noise + slow drift ──────
# Emissivity ε₂ = 0.72 → larger over-read
eps2          = 0.72
T_pyr2_ideal  = T_true_C * (1.0 / eps2) ** 0.25
noise2        = np.random.normal(0, 18, n)          # ±18 °C Gaussian noise
drift         = np.linspace(0, 35, n)               # slow 35 °C drift
spikes2       = np.zeros(n)
spike_idx2    = np.random.choice(n, size=int(n * 0.02), replace=False)
spikes2[spike_idx2] = np.random.uniform(150, 500, len(spike_idx2))
T_pyr2_raw    = T_pyr2_ideal + noise2 + drift + spikes2

# ── Thermocouple — clean reference with slight thermal lag (RC filter) ───────
alpha         = 0.08    # lag coefficient (higher = faster response)
T_tc          = np.zeros(n)
T_tc[0]       = T_true_C[0]
for i in range(1, n):
    T_tc[i] = T_tc[i-1] + alpha * (T_true_C[i] - T_tc[i-1])
T_tc += np.random.normal(0, 2, n)    # small thermocouple noise ±2 °C

print(f"\n  Pyrometer 1 raw   : mean={T_pyr1_raw.mean():.1f} °C, "
      f"spikes={len(spike_idx1)}")
print(f"  Pyrometer 2 raw   : mean={T_pyr2_raw.mean():.1f} °C, "
      f"spikes={len(spike_idx2)}, drift=35 °C")
print(f"  Thermocouple      : mean={T_tc.mean():.1f} °C, lag τ≈{1/alpha:.0f} steps")


# ─────────────────────────────────────────────────────────────────────────────
# ★ 5-ROW × 5-COLUMN RAW PREVIEW
#   Rows = time steps, Columns = [Time_s, Pyr1_raw, Pyr2_raw, TC, T_true]
# ─────────────────────────────────────────────────────────────────────────────

def make_df(start=0):
    """Return a 5-row preview DataFrame starting at index `start`."""
    idx = range(start, start + 5)
    return pd.DataFrame({
        "Time_s"   : np.round(time_s[list(idx)], 4),
        "Pyr1_raw" : np.round(T_pyr1_raw[list(idx)], 2),
        "Pyr2_raw" : np.round(T_pyr2_raw[list(idx)], 2),
        "TC_ref"   : np.round(T_tc[list(idx)], 2),
        "T_true"   : np.round(T_true_C[list(idx)], 2),
    }, index=[f"t{i}" for i in idx])

# Find a good preview window — around the hottest point
hot_start = max(0, T_true_C.argmax() - 2)

print()
print("=" * 65)
print("★ RAW DATA PREVIEW  (5 rows × 5 columns)")
print("  Columns: Time | Pyrometer1 | Pyrometer2 | Thermocouple | TrueTemp")
print("=" * 65)
df_raw = make_df(hot_start)
print(df_raw.to_string())
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DENOISING
#   Method A: Median filter — removes spikes without blurring edges
#   Method B: Gaussian-weighted moving average for residual noise
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Denoising  (spike removal + smoothing)")
print("=" * 65)

KERNEL = 7    # median filter window (odd number)

# Median filter kills impulse spikes cleanly
T_pyr1_med = medfilt(T_pyr1_raw, kernel_size=KERNEL)
T_pyr2_med = medfilt(T_pyr2_raw, kernel_size=KERNEL)

# Gaussian-weighted moving average (σ=3 steps) for residual noise
def gauss_smooth(signal, sigma=3):
    """Convolve signal with a Gaussian kernel of given sigma."""
    w  = int(4 * sigma + 1)
    x  = np.arange(-w, w + 1)
    k  = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return np.convolve(signal, k, mode="same")

T_pyr1_den = gauss_smooth(T_pyr1_med)
T_pyr2_den = gauss_smooth(T_pyr2_med)
T_tc_den   = gauss_smooth(T_tc, sigma=2)    # lighter smoothing on TC

# Count spikes removed (values changed by > 50 °C after median filter)
spikes_removed1 = (np.abs(T_pyr1_raw - T_pyr1_med) > 50).sum()
spikes_removed2 = (np.abs(T_pyr2_raw - T_pyr2_med) > 50).sum()

print(f"  ✓ Median filter (k={KERNEL}) applied to both pyrometers")
print(f"  ✓ Spikes removed — Pyr1: {spikes_removed1}, Pyr2: {spikes_removed2}")
print(f"  ✓ Gaussian smoothing (σ=3) applied")

print()
print("  ── AFTER DENOISING ──")
df_den = pd.DataFrame({
    "Time_s"   : np.round(time_s[hot_start:hot_start+5], 4),
    "Pyr1_den" : np.round(T_pyr1_den[hot_start:hot_start+5], 2),
    "Pyr2_den" : np.round(T_pyr2_den[hot_start:hot_start+5], 2),
    "TC_den"   : np.round(T_tc_den[hot_start:hot_start+5], 2),
    "T_true"   : np.round(T_true_C[hot_start:hot_start+5], 2),
}, index=[f"t{i}" for i in range(hot_start, hot_start+5)])
print(df_den.to_string())
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CALIBRATION
#   Goal: correct emissivity error in both pyrometers using thermocouple
#         as reference over a known calibration window.
#
#   Method: Linear regression  T_true ≈ a × T_pyr + b
#           Fit on first 20% of data (calibration region), apply to rest.
#           This is the "calibration hook" — replace with ML later.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 2 — Calibration  (emissivity correction via thermocouple)")
print("=" * 65)

cal_end = int(0.20 * n)    # first 20% = calibration window

def linear_cal(T_pyr, T_ref, cal_end):
    """
    Fit T_ref = a * T_pyr + b on calibration window.
    Return corrected full signal and coefficients.
    """
    x  = T_pyr[:cal_end]
    y  = T_ref[:cal_end]
    # Normal equations for least-squares line fit
    A  = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return a * T_pyr + b, a, b

T_pyr1_cal, a1, b1 = linear_cal(T_pyr1_den, T_tc_den, cal_end)
T_pyr2_cal, a2, b2 = linear_cal(T_pyr2_den, T_tc_den, cal_end)

# Remove the slow drift from Pyr2 using detrending on calibration window
drift_slope = np.polyfit(np.arange(n), T_pyr2_cal - T_tc_den, 1)[0]
T_pyr2_cal  = T_pyr2_cal - drift_slope * np.arange(n)

print(f"  ✓ Pyr1 calibration: T_cal = {a1:.4f} × T_raw + {b1:.2f}")
print(f"  ✓ Pyr2 calibration: T_cal = {a2:.4f} × T_raw + {b2:.2f}")
print(f"  ✓ Pyr2 drift correction: {drift_slope*1000:.3f} °C/1000-steps removed")

rmse1_before = np.sqrt(np.mean((T_pyr1_den - T_tc_den)**2))
rmse1_after  = np.sqrt(np.mean((T_pyr1_cal - T_tc_den)**2))
rmse2_before = np.sqrt(np.mean((T_pyr2_den - T_tc_den)**2))
rmse2_after  = np.sqrt(np.mean((T_pyr2_cal - T_tc_den)**2))
print(f"\n  RMSE vs TC reference — Pyr1: {rmse1_before:.1f} → {rmse1_after:.1f} °C")
print(f"  RMSE vs TC reference — Pyr2: {rmse2_before:.1f} → {rmse2_after:.1f} °C")

print()
print("  ── AFTER CALIBRATION ──")
df_cal = pd.DataFrame({
    "Time_s"   : np.round(time_s[hot_start:hot_start+5], 4),
    "Pyr1_cal" : np.round(T_pyr1_cal[hot_start:hot_start+5], 2),
    "Pyr2_cal" : np.round(T_pyr2_cal[hot_start:hot_start+5], 2),
    "TC_ref"   : np.round(T_tc_den[hot_start:hot_start+5], 2),
    "T_true"   : np.round(T_true_C[hot_start:hot_start+5], 2),
}, index=[f"t{i}" for i in range(hot_start, hot_start+5)])
print(df_cal.to_string())
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SENSOR FUSION
#   Combine Pyr1 and Pyr2 into one best-estimate signal.
#   Method: Inverse-variance weighting
#     w_i = 1 / σ_i²   where σ_i = local noise of each sensor
#     T_fused = (w1*T1 + w2*T2) / (w1 + w2)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 3 — Sensor Fusion  (inverse-variance weighted average)")
print("=" * 65)

# Estimate local noise using residual from smooth baseline
sigma1  = np.std(T_pyr1_cal - gauss_smooth(T_pyr1_cal, sigma=10)) + 1e-6
sigma2  = np.std(T_pyr2_cal - gauss_smooth(T_pyr2_cal, sigma=10)) + 1e-6
w1, w2  = 1 / sigma1**2, 1 / sigma2**2
T_fused = (w1 * T_pyr1_cal + w2 * T_pyr2_cal) / (w1 + w2)

rmse_fused = np.sqrt(np.mean((T_fused - T_tc_den)**2))
print(f"  ✓ Pyr1 weight: {w1/(w1+w2)*100:.1f}%   Pyr2 weight: {w2/(w1+w2)*100:.1f}%")
print(f"  ✓ RMSE of fused signal vs TC reference: {rmse_fused:.2f} °C")

print()
print("  ── AFTER FUSION (final 5×5 preview) ──")
df_fused = pd.DataFrame({
    "Time_s"   : np.round(time_s[hot_start:hot_start+5], 4),
    "Pyr1_cal" : np.round(T_pyr1_cal[hot_start:hot_start+5], 2),
    "Pyr2_cal" : np.round(T_pyr2_cal[hot_start:hot_start+5], 2),
    "T_fused"  : np.round(T_fused[hot_start:hot_start+5], 2),
    "T_true"   : np.round(T_true_C[hot_start:hot_start+5], 2),
}, index=[f"t{i}" for i in range(hot_start, hot_start+5)])
print(df_fused.to_string())
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SAVE CLEAN DATASET
#   Save the processed time-series as CSV so it can be used by other scripts.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 4 — Saving clean dataset → clean_pyrometer_data.csv")
print("=" * 65)

df_out = pd.DataFrame({
    "time_s"       : time_s,
    "pyr1_raw_C"   : T_pyr1_raw,
    "pyr2_raw_C"   : T_pyr2_raw,
    "tc_ref_C"     : T_tc,
    "pyr1_cal_C"   : T_pyr1_cal,
    "pyr2_cal_C"   : T_pyr2_cal,
    "fused_C"      : T_fused,
    "true_C"       : T_true_C,
})
df_out.to_csv("clean_pyrometer_data.csv", index=False)
print(f"  ✓ Saved {len(df_out)} rows × {len(df_out.columns)} columns")
print()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VISUALISATION  (4-panel dashboard)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 5 — Visualisation")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
#fig.suptitle("2-Pyrometer + Thermocouple Pipeline — Simulated AP&T Data\n"
            # "(Based on NIST Layer01 IN625 signal)", fontsize=13, fontweight="bold")

fig.suptitle("2-Pyrometer + Thermocouple Pipeline — Simulated AP&T Data\n"
             "(Based on NIST Layer03 IN625 signal)", fontsize=13, fontweight="bold")

# Panel A — Raw signals
ax = axes[0, 0]
ax.plot(time_s, T_pyr1_raw, alpha=0.5, color="steelblue",  lw=0.7, label="Pyr1 raw")
ax.plot(time_s, T_pyr2_raw, alpha=0.5, color="darkorange", lw=0.7, label="Pyr2 raw")
ax.plot(time_s, T_tc,       color="green", lw=1.2, label="Thermocouple")
ax.plot(time_s, T_true_C,   color="black", lw=1.0, ls="--", alpha=0.4, label="True temp")
ax.set_title("A  Raw signals")
ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8)
ax.set_xlabel("Time (s)")

# Panel B — After denoising
ax = axes[0, 1]
ax.plot(time_s, T_pyr1_den, color="steelblue",  lw=1.0, label="Pyr1 denoised")
ax.plot(time_s, T_pyr2_den, color="darkorange", lw=1.0, label="Pyr2 denoised")
ax.plot(time_s, T_tc_den,   color="green",      lw=1.2, label="TC denoised")
ax.plot(time_s, T_true_C,   color="black", lw=1.0, ls="--", alpha=0.4, label="True temp")
ax.set_title("B  After denoising (median + Gaussian)")
ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8)
ax.set_xlabel("Time (s)")

# Panel C — After calibration
ax = axes[1, 0]
ax.plot(time_s, T_pyr1_cal, color="steelblue",  lw=1.0, label=f"Pyr1 calibrated (RMSE={rmse1_after:.1f}°C)")
ax.plot(time_s, T_pyr2_cal, color="darkorange", lw=1.0, label=f"Pyr2 calibrated (RMSE={rmse2_after:.1f}°C)")
ax.plot(time_s, T_tc_den,   color="green",      lw=1.2, label="TC reference")
ax.plot(time_s, T_true_C,   color="black", lw=1.0, ls="--", alpha=0.4, label="True temp")
ax.set_title("C  After emissivity calibration (RMSE vs TC reference)")
ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8)
ax.set_xlabel("Time (s)")

# Panel D — Fused result vs TC reference
ax = axes[1, 1]
ax.plot(time_s, T_tc_den,  color="green",   lw=1.2, label="TC reference")
ax.plot(time_s, T_true_C,  color="black",   lw=1.0, ls="--", alpha=0.4, label="True temp")
ax.plot(time_s, T_fused,   color="crimson", lw=1.4, label=f"Fused (RMSE={rmse_fused:.2f}°C vs TC)")
ax.fill_between(time_s,
                T_fused - rmse_fused,
                T_fused + rmse_fused,
                color="crimson", alpha=0.15, label="±1 RMSE band")
ax.set_title("D  Final fused signal vs TC reference")
ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8)
ax.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("two_pyrometer_result.png", dpi=150, bbox_inches="tight")
print("  ✓ Figure saved → two_pyrometer_result.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("PIPELINE COMPLETE — SUMMARY")
print("=" * 65)
print(f"  Signals processed  : Pyr1, Pyr2, Thermocouple ({n} samples each)")
print(f"  Denoising          : median(k=7) + Gaussian(σ=3)")
print(f"  Spikes removed     : Pyr1={spikes_removed1}, Pyr2={spikes_removed2}")
print(f"  Calibration RMSE   : Pyr1 {rmse1_before:.1f}→{rmse1_after:.1f}°C | "
      f"Pyr2 {rmse2_before:.1f}→{rmse2_after:.1f}°C  (vs TC reference)")
print(f"  Fused RMSE         : {rmse_fused:.2f} °C vs TC reference")
print(f"  Output CSV         : clean_pyrometer_data.csv")
print()
print("  ★ When real AP&T data arrives:")
print("     Replace STEP 0 with: pd.read_csv('real_apt_data.csv')")
print("     Map columns to: time_s, T_pyr1, T_pyr2, T_tc")
print("     All other steps work unchanged.")
print("=" * 65)