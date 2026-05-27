"""
=============================================================================
podfam_summary.py  --  PODFAM Multi-File Summary for Thesis
=============================================================================
Thesis: Automation of Pyrometer Data Pre-processing
        for Metal Forming and Heat Treatment (AP&T / University West)

WHAT THIS DOES:
    Runs the full pipeline on ALL .pcd files in podfam_data folder
    and produces one clean summary plot + CSV table for the thesis.

    Shows:
        Plot 1 -- Hotspot temperature across all files (cooling curve)
        Plot 2 -- Noise reduction per file (consistent ~71%)
        Plot 3 -- Compression ratio and RMSE per file
        Plot 4 -- Summary table of all results

HOW TO RUN:
    python podfam_summary.py

OUTPUT:
    podfam_summary.png  -- thesis-ready summary plot
    podfam_summary.csv  -- results table
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denoise  import denoise_signal, count_spikes, noise_level
from compress import wavelet_compress, wavelet_reconstruct

# =============================================================================
# CONFIGURATION
# =============================================================================
PODFAM_FOLDER = r"C:\Users\sravy\OneDrive\Desktop\Thesis\podfam_data"

BG_S0  = 792.45
BG_S1  = 798.45
B1     = 2044.7
B2     = 0.83
HOTSPOT_N          = 100
ACTIVITY_THRESHOLD = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def read_pcd(filepath):
    rows = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                vals = [float(p) for p in parts]
                if len(vals) >= 6:
                    rows.append(vals)
            except ValueError:
                continue
    if not rows:
        return None, None, None
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 4].astype(np.float32), arr[:, 5].astype(np.float32)


def process_file(fpath):
    """Run full pipeline on one file. Returns dict of results."""
    fname = os.path.basename(fpath)

    # Parse time from filename e.g. 008.820.pcd -> 8.820
    try:
        t_val = float(fname.replace(".pcd", ""))
    except Exception:
        t_val = 0.0

    t_col, s0_raw, s1_raw = read_pcd(fpath)
    if s0_raw is None:
        return None

    n = len(s0_raw)

    # Step 1 -- Denoise
    s0_den    = denoise_signal(s0_raw, median_kernel=7, gauss_sigma=2.0)
    s1_den    = denoise_signal(s1_raw, median_kernel=7, gauss_sigma=2.0)
    noise_b   = noise_level(s0_raw)
    noise_a   = noise_level(s0_den)
    noise_red = (1 - noise_a / max(noise_b, 1e-6)) * 100
    spikes    = count_spikes(s0_raw, s0_den)

    # Step 2 -- Calibrate (hotspot)
    c0 = (s0_den - BG_S0).astype(np.float64)
    c1 = (s1_den - BG_S1).astype(np.float64)
    combined = c0 + c1
    top_idx  = np.argsort(combined)[-HOTSPOT_N:]
    c0t, c1t = c0[top_idx], c1[top_idx]
    act      = (c0t > ACTIVITY_THRESHOLD) & (c1t > ACTIVITY_THRESHOLD)

    if act.sum() > 0:
        ratio   = np.clip(c1t[act] / c0t[act], 1e-6, None)
        denom   = np.log(ratio) + B2
        denom   = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        T_hot   = np.clip(B1 / denom - 273.15, 0, 3000)
        T_med   = float(np.median(T_hot))
        T_max   = float(T_hot.max())
    else:
        T_med = T_max = 0.0

    # Full calibrated array for compression
    T_full = np.zeros(n, dtype=np.float32)
    active = (c0 > ACTIVITY_THRESHOLD) & (c1 > ACTIVITY_THRESHOLD)
    if active.sum() > 0:
        ratio2 = np.clip(c1[active] / c0[active], 1e-6, None)
        denom2 = np.log(ratio2) + B2
        denom2 = np.where(np.abs(denom2) < 1e-6, 1e-6, denom2)
        T_full[active] = np.clip(B1 / denom2 - 273.15, 0, 3000).astype(np.float32)

    # Step 3 -- Compress
    CHUNK      = min(n, 100000)
    cw         = wavelet_compress(T_full[:CHUNK], keep_fraction=0.10)
    T_rec      = wavelet_reconstruct(cw).astype(np.float32)[:CHUNK]
    orig_bytes = T_full[:CHUNK].nbytes
    comp_bytes = max(1, cw["nonzero"] * 8)
    ratio_comp = orig_bytes / comp_bytes
    rmse_comp  = float(np.sqrt(np.mean((T_full[:CHUNK] - T_rec) ** 2)))

    return {
        "file"        : fname,
        "time_s"      : t_val,
        "points"      : n,
        "noise_red_%" : round(noise_red, 1),
        "spikes"      : spikes,
        "T_hotspot_C" : round(T_med, 1),
        "T_max_C"     : round(T_max, 1),
        "comp_ratio"  : round(ratio_comp, 1),
        "comp_rmse_C" : round(rmse_comp, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN -- process all files
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("podfam_summary.py -- PODFAM Multi-File Summary")
print("=" * 65)

pcd_files = sorted(glob.glob(os.path.join(PODFAM_FOLDER, "*.pcd")))
print(f"\n  Found {len(pcd_files)} .pcd files in {PODFAM_FOLDER}")

if not pcd_files:
    print("  ERROR: No .pcd files found!")
    sys.exit(1)

results = []
for i, fpath in enumerate(pcd_files):
    fname = os.path.basename(fpath)
    print(f"  [{i+1:2d}/{len(pcd_files)}] Processing {fname} ...", end=" ")
    r = process_file(fpath)
    if r:
        results.append(r)
        print(f"T={r['T_hotspot_C']:.0f}°C  noise={r['noise_red_%']:.1f}%  "
              f"comp={r['comp_ratio']:.1f}x  RMSE={r['comp_rmse_C']:.1f}°C")
    else:
        print("FAILED")

df = pd.DataFrame(results).sort_values("time_s").reset_index(drop=True)

print()
print("=" * 65)
print("RESULTS TABLE")
print("=" * 65)
print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY STATISTICS")
print("=" * 65)
print(f"  Files processed     : {len(df)}")
print(f"  Time range          : {df['time_s'].min():.3f}s - {df['time_s'].max():.3f}s")
print(f"  Hotspot T range     : {df['T_hotspot_C'].min():.0f}°C - {df['T_hotspot_C'].max():.0f}°C")
print(f"  Avg noise reduction : {df['noise_red_%'].mean():.1f}%")
print(f"  Std noise reduction : {df['noise_red_%'].std():.2f}%")
print(f"  Avg comp ratio      : {df['comp_ratio'].mean():.1f}x")
print(f"  Avg comp RMSE       : {df['comp_rmse_C'].mean():.2f}°C")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────
print()
print("  Generating summary plot...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "PODFAM Pipeline -- AP&T Real Data Summary\n"
    f"Pipeline: Denoise → Two-Colour Calibration → Wavelet Compress  "
    f"|  {len(df)} files  |  {df['time_s'].min():.2f}s – {df['time_s'].max():.2f}s",
    fontsize=13, fontweight="bold"
)

times = df["time_s"].values
T_hot = df["T_hotspot_C"].values
noise = df["noise_red_%"].values
cratio = df["comp_ratio"].values
crmse  = df["comp_rmse_C"].values

# Panel A -- Hotspot temperature (cooling/heating curve)
ax = axes[0, 0]
ax.plot(times, T_hot, color="#e74c3c", lw=2.0, marker="o", ms=7,
        label="Hotspot T (median top 100 px)")
ax.fill_between(times, T_hot, alpha=0.15, color="#e74c3c")
for t, T, fname in zip(times, T_hot, df["file"]):
    ax.annotate(f"{T:.0f}°C", (t, T), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=7, color="#c0392b")
ax.set_title("A  Calibrated Hotspot Temperature Across Files")
ax.set_xlabel("File timestamp (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_facecolor("#fff5f5")
ax.set_ylim(0, max(T_hot) * 1.2)

# Panel B -- Noise reduction per file
ax = axes[0, 1]
bars = ax.bar(df["file"], noise, color="#3498db", alpha=0.85, edgecolor="white", width=0.6)
ax.axhline(noise.mean(), color="#1a5276", lw=2, ls="--",
           label=f"Mean = {noise.mean():.1f}%")
ax.set_title("B  Step 1 — Noise Reduction per File (%)")
ax.set_xlabel("File"); ax.set_ylabel("Noise reduction (%)")
ax.set_ylim(0, 100)
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#f0f8ff")
for bar, val in zip(bars, noise):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

# Panel C -- Compression RMSE per file
ax = axes[1, 0]
ax2 = ax.twinx()
bars2 = ax.bar(df["file"], cratio, color="#9b59b6", alpha=0.7,
               edgecolor="white", width=0.6, label="Compression ratio")
ax2.plot(df["file"], crmse, color="#e67e22", lw=2, marker="s", ms=6,
         label=f"RMSE (°C)")
ax.set_title("C  Step 3 — Compression Ratio & RMSE per File")
ax.set_xlabel("File"); ax.set_ylabel("Compression ratio (x)", color="#9b59b6")
ax2.set_ylabel("RMSE (°C)", color="#e67e22")
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0, max(cratio) * 1.5)
ax2.set_ylim(0, max(crmse) * 1.5)
ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#fdf0ff")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# Panel D -- Summary stats table
ax = axes[1, 1]
ax.axis("off")
summary = (
    f"PODFAM PIPELINE SUMMARY\n"
    f"{'─'*40}\n"
    f"Files processed      : {len(df)}\n"
    f"Time range           : {df['time_s'].min():.2f}s – {df['time_s'].max():.2f}s\n"
    f"\n"
    f"STEP 1 — DENOISING\n"
    f"  Method             : Median (k=7) + Gaussian\n"
    f"  Avg noise reduction: {noise.mean():.1f}%\n"
    f"  Std deviation      : ±{noise.std():.2f}%\n"
    f"\n"
    f"STEP 2 — CALIBRATION\n"
    f"  Formula            : T=2044.7/(ln(C1/C0)+0.83)\n"
    f"  Hotspot T range    : {T_hot.min():.0f}°C – {T_hot.max():.0f}°C\n"
    f"  Mean hotspot T     : {T_hot.mean():.0f}°C\n"
    f"\n"
    f"STEP 3 — COMPRESSION\n"
    f"  Method             : Wavelet (keep 10%)\n"
    f"  Avg ratio          : {cratio.mean():.1f}x\n"
    f"  Avg RMSE           : {crmse.mean():.2f}°C\n"
)
ax.text(0.05, 0.97, summary, transform=ax.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", alpha=0.9))

plt.tight_layout()
plt.savefig("podfam_summary.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> podfam_summary.png")

# Save CSV
df.to_csv("podfam_summary.csv", index=False)
print("  CSV  saved -> podfam_summary.csv")

print()
print("=" * 65)
print("DONE -- podfam_summary.png and podfam_summary.csv ready!")
print("=" * 65)
