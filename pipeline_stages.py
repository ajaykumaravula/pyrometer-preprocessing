"""
=============================================================================
pipeline_stages.py  —  See every stage output clearly in the terminal
=============================================================================
Chain:
    Layer01.mat
        ↓
    STAGE 0 — Raw data (°C)
        ↓  denoise.py
    STAGE 1 — Denoised
        ↓  calibrate.py  (takes DENOISED as input)
    STAGE 2 — Calibrated
        ↓  compress.py   (takes CALIBRATED as input)
    STAGE 3 — Compressed & Reconstructed

At every stage you see the SAME 5 rows so you can track changes.
=============================================================================
"""

import numpy as np
import scipy.io as sio
import pandas as pd
import sys
sys.path.insert(0, ".")

from denoise   import denoise_signal
from calibrate import linear_calibration, remove_drift
from compress  import wavelet_compress, wavelet_reconstruct

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
#DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"
DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"

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

# Thermocouple reference
T_tc = np.zeros(n)
T_tc[0] = T_raw[0]
for i in range(1, n):
    T_tc[i] = T_tc[i-1] + 0.08 * (T_raw[i] - T_tc[i-1])
T_tc += np.random.default_rng(42).normal(0, 2, n)

# Pick 5 rows around the hottest point — same rows used throughout
HOT = max(0, T_raw.argmax() - 2)
idx = list(range(HOT, HOT + 5))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — RAW
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STAGE 0 — RAW DATA  (direct from Layer01.mat sensor)")
print("=" * 65)
df0 = pd.DataFrame({
    "Time_s"  : np.round(time_s[idx], 4),
    "Raw_C"   : np.round(T_raw[idx], 2),
    "TC_ref"  : np.round(T_tc[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df0.to_string())
print(f"\n  Range : {T_raw.min():.1f} – {T_raw.max():.1f} °C")
print(f"  Note  : Raw has spikes and emissivity error")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — DENOISE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STAGE 1 — AFTER DENOISING  [denoise.py]")
print("  Input  : Raw_C  (from Stage 0)")
print("  Output : Denoised_C")
print("=" * 65)

T_den = denoise_signal(T_raw, median_kernel=7, gauss_sigma=3.0)

df1 = pd.DataFrame({
    "Time_s"      : np.round(time_s[idx], 4),
    "Raw_C"       : np.round(T_raw[idx], 2),       # what went IN
    "Denoised_C"  : np.round(T_den[idx], 2),        # what came OUT
    "Change"      : np.round(T_den[idx] - T_raw[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df1.to_string())
print(f"\n  Spikes removed and signal smoothed")
print(f"  Denoised_C is now the INPUT to Stage 2")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CALIBRATE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STAGE 2 — AFTER CALIBRATION  [calibrate.py]")
print("  Input  : Denoised_C  (from Stage 1)  ← NOT raw")
print("  Output : Calibrated_C")
print("=" * 65)

T_cal, coeffs = linear_calibration(T_den, T_tc, cal_fraction=0.20)
T_cal         = remove_drift(T_cal, T_tc)

df2 = pd.DataFrame({
    "Time_s"       : np.round(time_s[idx], 4),
    "Denoised_C"   : np.round(T_den[idx], 2),       # what went IN
    "Calibrated_C" : np.round(T_cal[idx], 2),        # what came OUT
    "TC_ref"       : np.round(T_tc[idx], 2),          # reference target
    "Error_vs_TC"  : np.round(T_cal[idx] - T_tc[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df2.to_string())
print(f"\n  Formula : T_cal = {coeffs['a']:.5f} × Denoised + {coeffs['b']:.2f}")
print(f"  Calibrated_C is now the INPUT to Stage 3")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — COMPRESS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STAGE 3 — AFTER COMPRESSION  [compress.py]")
print("  Input  : Calibrated_C  (from Stage 2)  ← NOT raw, NOT denoised")
print("  Output : Reconstructed_C")
print("=" * 65)

comp        = wavelet_compress(T_cal, keep_fraction=0.10)
T_recon     = wavelet_reconstruct(comp)

orig_bytes  = T_cal.nbytes
comp_bytes  = max(1, comp["nonzero"] * 8)
ratio       = orig_bytes / comp_bytes
rmse_comp   = float(np.sqrt(np.mean((T_cal - T_recon) ** 2)))

df3 = pd.DataFrame({
    "Time_s"         : np.round(time_s[idx], 4),
    "Calibrated_C"   : np.round(T_cal[idx], 2),     # what went IN
    "Reconstructed_C": np.round(T_recon[idx], 2),    # what came OUT
    "Error"          : np.round(T_recon[idx] - T_cal[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df3.to_string())
print(f"\n  Compression : {ratio:.1f}× smaller")
print(f"  RMSE error  : {rmse_comp:.2f} °C  (small = good)")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL — ALL STAGES SIDE BY SIDE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("★ ALL STAGES SIDE BY SIDE — same 5 rows throughout")
print("=" * 65)
df_all = pd.DataFrame({
    "Time_s"          : np.round(time_s[idx], 4),
    "Raw_C"           : np.round(T_raw[idx], 2),
    "Denoised_C"      : np.round(T_den[idx], 2),
    "Calibrated_C"    : np.round(T_cal[idx], 2),
    "Reconstructed_C" : np.round(T_recon[idx], 2),
    "TC_ref"          : np.round(T_tc[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df_all.to_string())

print()
print("  Column guide:")
print("  Raw_C          — sensor output (has spikes + emissivity error)")
print("  Denoised_C     — after spike removal and smoothing")
print("  Calibrated_C   — after emissivity correction using thermocouple")
print("  Reconstructed_C— after compress + decompress (ready for storage)")
print("  TC_ref         — thermocouple reference (ground truth)")
print()
print("  Pipeline flow:")
print("  Raw_C → [denoise.py] → Denoised_C")
print("  Denoised_C → [calibrate.py] → Calibrated_C")
print("  Calibrated_C → [compress.py] → Reconstructed_C")
print("=" * 65)
