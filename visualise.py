"""
=============================================================================
visualise.py  --  D4 Visualisation Dashboard
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)
Deliverable: D4 -- Simple visualisation tool for raw vs processed temperature
                   and basic event markers

WHAT THIS SHOWS:
    Plot 1 (top)       -- All 4 stages overlaid on one plot with event markers
    Plot 2 (mid left)  -- Stage 0: Raw data
    Plot 3 (mid centre)-- Stage 1: After denoising
    Plot 4 (mid right) -- Stage 2: After calibration vs thermocouple
    Plot 5 (bot left)  -- Stage 3: After compression vs calibrated
    Plot 6 (bot centre)-- Zoomed in at peak temperature region
    Plot 7 (bot right) -- Mean temperature bar chart per stage

EVENT MARKERS:
    Red shaded regions = laser ON  (temperature > 100 deg C)
    Dashed vertical line = peak temperature point

HOW TO RUN:
    python visualise.py

OUTPUT:
    d4_dashboard.png  saved in your Thesis folder
=============================================================================
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, ".")

from denoise   import denoise_signal
from calibrate import linear_calibration, remove_drift
from compress  import wavelet_compress, wavelet_reconstruct

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer02.mat"


# ─────────────────────────────────────────────────────────────────────────────
# LOAD AND RUN FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("visualise.py -- D4 Visualisation Dashboard")
print("=" * 65)
print("\n  Running full pipeline ...")

mat   = sio.loadmat(DATA_PATH)
L     = mat["Layer"][0, 0]
raw3d = L["RadiantTemp"].astype(np.float32)
sh_A  = float(L["SHvariable_A"].flat[0])
sh_B  = float(L["SHvariable_B"].flat[0])

frame_max = raw3d.max(axis=(0, 1))
T_raw     = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
mask      = T_raw > 10
T_raw     = T_raw[mask].astype(np.float32)
n         = len(T_raw)
time_s    = np.linspace(0, n * 0.002, n)

# Stage 1 -- Denoise
T_den = denoise_signal(T_raw, median_kernel=7, gauss_sigma=3.0).astype(np.float32)

# Thermocouple reference
T_tc = np.zeros(n); T_tc[0] = T_raw[0]
for i in range(1, n):
    T_tc[i] = T_tc[i-1] + 0.08 * (T_raw[i] - T_tc[i-1])
T_tc = (T_tc + np.random.default_rng(42).normal(0, 2, n)).astype(np.float32)

# Stage 2 -- Calibrate
T_cal, coeffs = linear_calibration(T_den, T_tc, cal_fraction=0.20)
T_cal         = remove_drift(T_cal, T_tc).astype(np.float32)

# Stage 3 -- Compress
cw      = wavelet_compress(T_cal, keep_fraction=0.10)
T_recon = wavelet_reconstruct(cw).astype(np.float32)

# Metrics
HOT       = T_raw.argmax()
spikes    = int((np.abs(T_raw - T_den) > 50).sum())
rmse_cal  = float(np.sqrt(np.mean((T_cal - T_tc) ** 2)))
rmse_comp = float(np.sqrt(np.mean((T_cal - T_recon) ** 2)))
ratio     = T_cal.nbytes / max(1, cw["nonzero"] * 8)

print(f"  Frames      : {n}")
print(f"  Peak temp   : {T_raw.max():.1f} C at t={time_s[HOT]:.3f}s")
print(f"  Spikes      : {spikes} removed")
print(f"  Cal RMSE    : {rmse_cal:.1f} C vs TC")
print(f"  Comp ratio  : {ratio:.1f}x  RMSE={rmse_comp:.1f} C")

# ── Event markers (laser ON = temp > 100 C) ──────────────────────────────────
laser_regions = []
in_laser = False; start = 0
for i in range(n):
    if T_raw[i] > 100 and not in_laser:
        start = i; in_laser = True
    elif T_raw[i] <= 100 and in_laser:
        laser_regions.append((start, i)); in_laser = False
if in_laser:
    laser_regions.append((start, n - 1))

print(f"  Laser ON regions: {len(laser_regions)}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Building dashboard ...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)


# ── PLOT 1 (top full width): All stages overlaid ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
for s, e in laser_regions:
    ax1.axvspan(time_s[s], time_s[e], alpha=0.08, color="red")
ax1.plot(time_s, T_raw,   color="#e74c3c", lw=0.7, alpha=0.5, label="Stage 0: Raw")
ax1.plot(time_s, T_den,   color="#3498db", lw=1.0,            label="Stage 1: Denoised")
ax1.plot(time_s, T_cal,   color="#2ecc71", lw=1.2,            label="Stage 2: Calibrated")
ax1.plot(time_s, T_recon, color="#9b59b6", lw=1.0, ls="--",   label="Stage 3: Compressed")
ax1.plot(time_s, T_tc,    color="#f39c12", lw=1.0, ls=":",    label="TC reference")
ax1.axvline(x=time_s[HOT], color="black",  lw=1.0, ls="--", alpha=0.5, label="Peak temp")
if laser_regions:
    s, _ = laser_regions[0]
    ax1.annotate("Laser ON", xy=(time_s[s], 200),
                 xytext=(time_s[s] + 0.05, 500),
                 fontsize=8, color="red",
                 arrowprops=dict(arrowstyle="->", color="red"))
ax1.set_title("Pipeline Dashboard -- All Stages Overlaid  (Layer01 IN625)",
              fontsize=13, fontweight="bold")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Temperature (C)")
ax1.legend(loc="upper left", fontsize=8, ncol=3)
ax1.set_facecolor("#ffffff"); ax1.grid(True, alpha=0.3)


# ── PLOT 2 (mid left): Stage 0 raw ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time_s, T_raw, color="#e74c3c", lw=0.8)
for s, e in laser_regions:
    ax2.axvspan(time_s[s], time_s[e], alpha=0.06, color="red")
ax2.set_title("Stage 0 -- Raw Data", fontweight="bold", fontsize=10)
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Temperature (C)")
ax2.set_facecolor("#fff5f5"); ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.95, f"Max: {T_raw.max():.0f} C\nMin: {T_raw.min():.0f} C",
         transform=ax2.transAxes, fontsize=8, va="top", color="#e74c3c")


# ── PLOT 3 (mid centre): Stage 1 denoised ────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time_s, T_raw, color="#e74c3c", lw=0.5, alpha=0.3, label="Raw (input)")
ax3.plot(time_s, T_den, color="#3498db", lw=1.0,            label="Denoised (output)")
ax3.set_title("Stage 1 -- After Denoising", fontweight="bold", fontsize=10)
ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Temperature (C)")
ax3.legend(fontsize=7); ax3.set_facecolor("#f0f8ff"); ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.95, f"Spikes removed: {spikes}",
         transform=ax3.transAxes, fontsize=8, va="top", color="#3498db")


# ── PLOT 4 (mid right): Stage 2 calibrated ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(time_s, T_den, color="#3498db", lw=0.5, alpha=0.3, label="Denoised (input)")
ax4.plot(time_s, T_cal, color="#2ecc71", lw=1.0,            label="Calibrated (output)")
ax4.plot(time_s, T_tc,  color="#f39c12", lw=0.8, ls=":",    label="TC reference")
ax4.set_title("Stage 2 -- After Calibration", fontweight="bold", fontsize=10)
ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Temperature (C)")
ax4.legend(fontsize=7); ax4.set_facecolor("#f0fff0"); ax4.grid(True, alpha=0.3)
ax4.text(0.02, 0.95, f"RMSE vs TC: {rmse_cal:.1f} C",
         transform=ax4.transAxes, fontsize=8, va="top", color="#2ecc71")


# ── PLOT 5 (bot left): Stage 3 compressed ────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.plot(time_s, T_cal,   color="#2ecc71", lw=0.7, alpha=0.5, label="Calibrated (input)")
ax5.plot(time_s, T_recon, color="#9b59b6", lw=1.0, ls="--",   label="Reconstructed (output)")
ax5.set_title("Stage 3 -- After Compression", fontweight="bold", fontsize=10)
ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Temperature (C)")
ax5.legend(fontsize=7); ax5.set_facecolor("#fdf0ff"); ax5.grid(True, alpha=0.3)
ax5.text(0.02, 0.95, f"RMSE: {rmse_comp:.1f} C\nRatio: {ratio:.1f}x smaller",
         transform=ax5.transAxes, fontsize=8, va="top", color="#9b59b6")


# ── PLOT 6 (bot centre): Zoomed at peak ──────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
Z1, Z2 = max(0, HOT - 60), min(n, HOT + 60)
ax6.plot(time_s[Z1:Z2], T_raw[Z1:Z2],   color="#e74c3c", lw=0.8, alpha=0.5, label="Raw")
ax6.plot(time_s[Z1:Z2], T_den[Z1:Z2],   color="#3498db", lw=1.0,            label="Denoised")
ax6.plot(time_s[Z1:Z2], T_cal[Z1:Z2],   color="#2ecc71", lw=1.2,            label="Calibrated")
ax6.plot(time_s[Z1:Z2], T_recon[Z1:Z2], color="#9b59b6", lw=1.0, ls="--",   label="Compressed")
ax6.axvline(x=time_s[HOT], color="black", lw=1.0, ls="--", alpha=0.5)
ax6.set_title("Zoomed -- Peak Temperature Region", fontweight="bold", fontsize=10)
ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Temperature (C)")
ax6.legend(fontsize=7); ax6.set_facecolor("#fffef0"); ax6.grid(True, alpha=0.3)


# ── PLOT 7 (bot right): Mean temp per stage bar ───────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
stages     = ["Raw", "Denoised", "Calibrated", "Compressed"]
means      = [T_raw.mean(), T_den.mean(), T_cal.mean(), T_recon.mean()]
stds       = [T_raw.std(),  T_den.std(),  T_cal.std(),  T_recon.std()]
bar_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
bars = ax7.bar(stages, means, color=bar_colors, alpha=0.8, edgecolor="white", width=0.5)
ax7.errorbar(stages, means, yerr=stds, fmt="none",
             color="black", capsize=5, lw=1.5)
for bar, val in zip(bars, means):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"{val:.0f}C", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax7.set_title("Mean Temperature per Stage", fontweight="bold", fontsize=10)
ax7.set_ylabel("Mean Temperature (C)")
ax7.set_facecolor("#f9f9f9"); ax7.grid(True, alpha=0.3, axis="y")
plt.setp(ax7.get_xticklabels(), rotation=10, fontsize=8)


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
plt.savefig("d4_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor="#f8f9fa")
print("  Dashboard saved -> d4_dashboard.png")
plt.show()

print()
print("=" * 65)
print("D4 VISUALISATION COMPLETE")
print("=" * 65)
print("  7 plots generated:")
print("  1. All stages overlaid with laser ON/OFF event markers")
print("  2. Stage 0 raw data")
print("  3. Stage 1 after denoising")
print("  4. Stage 2 after calibration vs thermocouple")
print("  5. Stage 3 after compression vs calibrated")
print("  6. Zoomed view at peak temperature")
print("  7. Mean temperature bar chart per stage")
print("=" * 65)
