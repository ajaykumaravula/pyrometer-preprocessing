"""
=============================================================================
podfam_all_files.py  --  Run PODFAM pipeline on ALL 10 files
=============================================================================
Runs all classical methods on all 10 PODFAM files and saves:
    - Individual result plot for each file
    - Combined summary CSV
    - Combined summary plot

HOW TO RUN:
    python podfam_all_files.py

FILES TESTED:
    000.280.pcd, 002.030.pcd, 003.080.pcd, 004.060.pcd, 005.110.pcd
    006.090.pcd, 007.000.pcd, 008.820.pcd, 009.030.pcd, 010.010.pcd
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import medfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error
import os
import sys
sys.path.insert(0, ".")

from compress import wavelet_compress, wavelet_reconstruct, \
                     svd_compress, svd_reconstruct

# =============================================================================
# CONFIGURATION -- Change this folder path to your PODFAM data location
# =============================================================================
PODFAM_FOLDER = r"C:\Users\sravy\OneDrive\Desktop\Thesis\podfam_data"

PCD_FILES = [
    "000.280.pcd",
    "002.030.pcd",
    "003.080.pcd",
    "004.060.pcd",
    "005.110.pcd",
    "006.090.pcd",
    "007.000.pcd",
    "008.820.pcd",
    "009.030.pcd",
    "010.010.pcd",
]

# Two-colour calibration constants (Karthikeyan, AP&T)
B1    = 2044.7
B2    = 0.83
BG_S0 = 792.45
BG_S1 = 798.45
TOP_N = 100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def noise_level(sig):
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

def moving_average(sig, k=7):
    return np.convolve(sig, np.ones(k)/k, mode="same").astype(np.float32)

def median_filt(sig, k=7):
    return medfilt(sig.astype(np.float64), kernel_size=k).astype(np.float32)

def savgol(sig, w=11, p=3):
    return savgol_filter(sig.astype(np.float64),
                         window_length=w, polyorder=p).astype(np.float32)

def gauss_filt(sig, sigma=3.0):
    return gaussian_filter1d(sig.astype(np.float64),
                             sigma=sigma).astype(np.float32)

def kalman_filt(sig, Q=1e-3, R=10.0):
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
    d  = np.diff(sig.astype(np.float64))
    dq = np.clip(np.round(d*100), -32767, 32767).astype(np.int16)
    return {"first": float(sig[0]), "deltas": dq, "scale": 100.0}

def delta_decode(enc):
    d = enc["deltas"].astype(np.float64) / enc["scale"]
    return np.concatenate([[enc["first"]],
                            np.cumsum(d)+enc["first"]]).astype(np.float32)

def two_colour_temp(s0_corr, s1_corr):
    ratio = s1_corr / np.maximum(s0_corr, 1e-6)
    ratio = np.clip(ratio, 1e-6, None)
    T_K   = B1 / (np.log(ratio) + B2)
    return T_K - 273.15

def load_pcd(filepath):
    rows = []
    with open(filepath, 'r') as f:
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
    return df

def process_file(pcd_path):
    """Process one PODFAM file through all classical methods."""

    fname = os.path.basename(pcd_path)
    print(f"\n  Processing: {fname}")

    # Load
    df     = load_pcd(pcd_path)
    S0_raw = df['sensor0'].values.astype(np.float32)
    S1_raw = df['sensor1'].values.astype(np.float32)
    time_s = df['time_s'].values
    n      = len(S0_raw)
    raw_bytes = S0_raw.nbytes

    # Background correction
    S0_corr = S0_raw - BG_S0
    S1_corr = S1_raw - BG_S1

    # ── DENOISING ────────────────────────────────────────────
    noise_raw = noise_level(S0_raw)
    T_ma      = moving_average(S0_raw, 7)
    T_med     = median_filt(S0_raw, 7)
    T_sg      = savgol(S0_raw, 11, 3)
    T_gauss   = gauss_filt(S0_raw, 3.0)
    T_kalman  = kalman_filt(S0_raw, 1e-3, 10.0)

    den_results = {}
    for name, sig in [
        ("Moving Average", T_ma),
        ("Median Filter",  T_med),
        ("Savitzky-Golay", T_sg),
        ("Gaussian",       T_gauss),
        ("Kalman",         T_kalman),
    ]:
        nl  = noise_level(sig)
        red = (1 - nl/noise_raw)*100
        spk = int((np.abs(S0_raw - sig) > 5).sum())
        den_results[name] = {"noise":round(nl,2),
                             "reduction":round(red,1),
                             "spikes":spk}

    # Primary denoised = Median
    T_den = T_med.copy()

    # ── CALIBRATION ──────────────────────────────────────────
    combined = S0_corr + S1_corr
    top_idx  = np.argsort(combined)[-TOP_N:]
    T_top    = two_colour_temp(S0_corr[top_idx], S1_corr[top_idx])
    T_top    = T_top[np.isfinite(T_top)]
    T_median_pool = float(np.median(T_top)) if len(T_top) > 0 else 0
    T_max_pool    = float(np.max(T_top))    if len(T_top) > 0 else 0
    T_min_pool    = float(np.min(T_top))    if len(T_top) > 0 else 0

    # ── COMPRESSION ──────────────────────────────────────────
    T_comp    = T_den.copy()
    comp_results = {}

    # Delta
    enc_delta  = delta_encode(T_comp)
    T_delta    = delta_decode(enc_delta)
    n2         = min(len(T_comp), len(T_delta))
    rmse_delta = float(np.sqrt(mean_squared_error(T_comp[:n2], T_delta[:n2])))
    ratio_delta = raw_bytes / (8 + enc_delta["deltas"].nbytes)
    comp_results["Delta Encoding"] = {"ratio":round(ratio_delta,1),
                                       "rmse":round(rmse_delta,4)}

    # Downsample
    T_ds    = np.interp(np.arange(n),
                        np.arange(0,n,2)[:len(T_comp[::2])],
                        T_comp[::2]).astype(np.float32)
    rmse_ds = float(np.sqrt(mean_squared_error(T_comp, T_ds)))
    comp_results["Downsampling 2x"] = {"ratio":2.0,"rmse":round(rmse_ds,2)}

    # SVD
    chunk = min(2048, n)
    if chunk >= 32:
        s2d     = T_comp[:chunk].reshape(-1,32).astype(np.float64)
        cs      = svd_compress(s2d, rank=10)
        rs      = svd_reconstruct(cs)
        T_svd   = np.interp(np.arange(n),
                            np.linspace(0,n-1,rs.ravel().shape[0]),
                            rs.ravel()).astype(np.float32)
        rmse_svd  = float(np.sqrt(mean_squared_error(T_comp, T_svd)))
        ratio_svd = s2d.nbytes/(cs["U"].nbytes+cs["S"].nbytes+cs["Vt"].nbytes)
        comp_results["SVD rank=10"] = {"ratio":round(ratio_svd,1),
                                        "rmse":round(rmse_svd,2)}

    # Wavelet
    cw        = wavelet_compress(T_comp, keep_fraction=0.10)
    T_wav     = wavelet_reconstruct(cw).astype(np.float32)
    n3        = min(len(T_comp), len(T_wav))
    rmse_wav  = float(np.sqrt(mean_squared_error(T_comp[:n3], T_wav[:n3])))
    ratio_wav = raw_bytes / max(1, cw["nonzero"]*8)
    comp_results["Wavelet 10%"] = {"ratio":round(ratio_wav,1),
                                    "rmse":round(rmse_wav,2)}

    # ── PLOT ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"PODFAM Pipeline -- {fname}", fontsize=12, fontweight="bold")

    ax = axes[0,0]
    ax.plot(time_s, S0_raw,  color="red",  lw=0.4, alpha=0.5, label="Raw S0")
    ax.plot(time_s, T_ma,    color="#3498db", lw=0.8, label="MovAvg")
    ax.plot(time_s, T_med,   color="#2ecc71", lw=0.9, label="Median")
    ax.plot(time_s, T_sg,    color="#9b59b6", lw=0.8, ls="--", label="SavGol")
    ax.plot(time_s, T_gauss, color="#f39c12", lw=0.8, label="Gaussian")
    ax.plot(time_s, T_kalman,color="#1abc9c", lw=0.8, ls=":", label="Kalman")
    ax.set_title("Denoising -- All 5 Methods")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Counts")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    methods_n = ["Raw","MovAvg","Median","SavGol","Gaussian","Kalman"]
    noise_v   = [noise_raw] + [den_results[k]["noise"] for k in
                 ["Moving Average","Median Filter","Savitzky-Golay",
                  "Gaussian","Kalman"]]
    colors_n  = ["#e74c3c","#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c"]
    bars = ax.bar(methods_n, noise_v, color=colors_n, alpha=0.85)
    for bar, val in zip(bars, noise_v):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Noise Level (lower=better)")
    ax.set_ylabel("Noise std dev"); ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    ax = axes[1,0]
    ax.plot(time_s, T_comp,  color="black",   lw=1.0, label="Denoised input")
    ax.plot(time_s, T_delta, color="#e74c3c",  lw=0.8,
            label=f"Delta {ratio_delta:.1f}x")
    ax.plot(time_s, T_ds,    color="#3498db",  lw=0.8, ls="--",
            label="Downsamp 2x")
    if "SVD rank=10" in comp_results:
        ax.plot(time_s, T_svd, color="#9b59b6", lw=0.8, ls="-.",
                label=f"SVD {ratio_svd:.1f}x")
    ax.plot(time_s[:n3], T_wav[:n3], color="#2ecc71", lw=0.9, ls=":",
            label=f"Wavelet {ratio_wav:.1f}x")
    ax.set_title("Compression -- All 4 Methods")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Counts")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    comp_names  = list(comp_results.keys())
    comp_ratios = [comp_results[k]["ratio"] for k in comp_names]
    comp_rmses  = [comp_results[k]["rmse"]  for k in comp_names]
    colors_c    = ["#e74c3c","#3498db","#9b59b6","#2ecc71"]
    ax.scatter(comp_ratios, comp_rmses, c=colors_c[:len(comp_names)],
               s=150, zorder=5, edgecolors="black", lw=0.8)
    for name, x, y in zip(comp_names, comp_ratios, comp_rmses):
        ax.annotate(name, (x,y), textcoords="offset points",
                    xytext=(6,5), fontsize=8)
    ax.axvline(x=4.0, color="green", ls="--", lw=1.2, label="4x target")
    ax.set_title("RMSE vs Ratio (bottom-right=best)")
    ax.set_xlabel("Compression ratio"); ax.set_ylabel("RMSE")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = fname.replace(".pcd","_result.png")
    plt.savefig(out_name, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    Saved --> {out_name}")

    return {
        "file"            : fname,
        "points"          : n,
        "noise_raw"       : round(noise_raw, 2),
        "den_results"     : den_results,
        "T_melt_median"   : round(T_median_pool, 1),
        "T_melt_max"      : round(T_max_pool, 1),
        "T_melt_min"      : round(T_min_pool, 1),
        "comp_results"    : comp_results,
    }

# =============================================================================
# RUN ALL FILES
# =============================================================================
print("=" * 65)
print("podfam_all_files.py -- All 10 PODFAM Files")
print("=" * 65)

all_results = []
for fname in PCD_FILES:
    fpath = os.path.join(PODFAM_FOLDER, fname)
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {fname}")
        continue
    result = process_file(fpath)
    all_results.append(result)

print()
print("=" * 65)
print(f"PROCESSED {len(all_results)} files")
print("=" * 65)

if not all_results:
    print("No files processed. Check PODFAM_FOLDER path.")
    sys.exit(1)

# =============================================================================
# COMBINED SUMMARY TABLE
# =============================================================================
rows = []
for r in all_results:
    row = {"File": r["file"], "Points": r["points"],
           "Noise_Raw": r["noise_raw"],
           "T_Melt_Median_C": r["T_melt_median"],
           "T_Melt_Max_C":    r["T_melt_max"],
           "T_Melt_Min_C":    r["T_melt_min"]}
    # Add denoising results
    for method, vals in r["den_results"].items():
        key = method.replace(" ","_")
        row[f"Den_{key}_noise"]     = vals["noise"]
        row[f"Den_{key}_reduction"] = vals["reduction"]
    # Add compression results
    for method, vals in r["comp_results"].items():
        key = method.replace(" ","_")
        row[f"Comp_{key}_ratio"] = vals["ratio"]
        row[f"Comp_{key}_rmse"]  = vals["rmse"]
    rows.append(row)

df_all = pd.DataFrame(rows)
df_all.to_csv("podfam_all_results.csv", index=False)
print("\n  Saved --> podfam_all_results.csv")

# =============================================================================
# PRINT SUMMARY
# =============================================================================
print()
print("=" * 65)
print("SUMMARY TABLE -- All Files")
print("=" * 65)

print("\n  DENOISING -- Median Filter (primary method):")
print(f"  {'File':<20} {'Noise Raw':>10} {'Noise Med':>10} {'Reduction%':>12} {'Spikes':>8}")
print("  " + "-"*62)
for r in all_results:
    med = r["den_results"]["Median Filter"]
    print(f"  {r['file']:<20} {r['noise_raw']:>10.2f} "
          f"{med['noise']:>10.2f} {med['reduction']:>12.1f} "
          f"{med['spikes']:>8}")

print("\n  CALIBRATION -- Two-colour formula:")
print(f"  {'File':<20} {'T_Melt_Min':>12} {'T_Melt_Median':>14} {'T_Melt_Max':>12}")
print("  " + "-"*60)
for r in all_results:
    print(f"  {r['file']:<20} {r['T_melt_min']:>12.1f} "
          f"{r['T_melt_median']:>14.1f} {r['T_melt_max']:>12.1f}")

print("\n  COMPRESSION -- Wavelet 10% (primary method):")
print(f"  {'File':<20} {'Ratio':>8} {'RMSE':>8}")
print("  " + "-"*38)
for r in all_results:
    if "Wavelet 10%" in r["comp_results"]:
        wav = r["comp_results"]["Wavelet 10%"]
        print(f"  {r['file']:<20} {wav['ratio']:>8.1f} {wav['rmse']:>8.2f}")

# Mean values
med_reductions = [r["den_results"]["Median Filter"]["reduction"]
                  for r in all_results]
wav_ratios = [r["comp_results"]["Wavelet 10%"]["ratio"]
              for r in all_results if "Wavelet 10%" in r["comp_results"]]
wav_rmses  = [r["comp_results"]["Wavelet 10%"]["rmse"]
              for r in all_results if "Wavelet 10%" in r["comp_results"]]
melt_temps = [r["T_melt_median"] for r in all_results]

print()
print("  MEAN VALUES:")
print(f"    Median noise reduction : {np.mean(med_reductions):.1f}% "
      f"(std {np.std(med_reductions):.2f}%)")
print(f"    Melt pool temperature  : {np.mean(melt_temps):.1f} C "
      f"(range {min(melt_temps):.1f}--{max(melt_temps):.1f} C)")
print(f"    Wavelet compression    : {np.mean(wav_ratios):.1f}x ratio, "
      f"{np.mean(wav_rmses):.2f} C RMSE")

# =============================================================================
# COMBINED SUMMARY PLOT
# =============================================================================
print()
print("  Generating summary plot ...")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#f8f9fa")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

files_short = [r["file"].replace(".pcd","") for r in all_results]
x = np.arange(len(all_results))

# Panel A: Noise reduction all methods
ax = fig.add_subplot(gs[0, :2])
methods_den = ["Moving Average","Median Filter","Savitzky-Golay",
               "Gaussian","Kalman"]
colors_den  = ["#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c"]
width = 0.15
for i, (method, color) in enumerate(zip(methods_den, colors_den)):
    reductions = [r["den_results"][method]["reduction"] for r in all_results]
    ax.bar(x + i*width, reductions, width, label=method,
           color=color, alpha=0.85)
ax.set_title("Denoising -- Noise Reduction % (all methods, all files)",
             fontweight="bold")
ax.set_xticks(x + width*2)
ax.set_xticklabels(files_short, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Noise Reduction (%)")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#ffffff")

# Panel B: Melt pool temperature
ax = fig.add_subplot(gs[0, 2])
temps_med = [r["T_melt_median"] for r in all_results]
temps_max = [r["T_melt_max"]    for r in all_results]
temps_min = [r["T_melt_min"]    for r in all_results]
ax.fill_between(x, temps_min, temps_max, alpha=0.3, color="#e74c3c",
                label="Min-Max range")
ax.plot(x, temps_med, "o-", color="#e74c3c", lw=2.0, ms=6,
        label="Median temp")
ax.axhline(y=1500, color="green", ls="--", lw=1.2, label="Expected min")
ax.axhline(y=2000, color="orange",ls="--", lw=1.2, label="Expected max")
ax.set_title("Melt Pool Temperature\n(Two-Colour Formula)",
             fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(files_short, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_facecolor("#ffffff")

# Panel C: Compression ratio all methods
ax = fig.add_subplot(gs[1, :2])
methods_comp = ["Delta Encoding","Downsampling 2x","SVD rank=10","Wavelet 10%"]
colors_comp  = ["#e74c3c","#3498db","#9b59b6","#2ecc71"]
width = 0.2
for i, (method, color) in enumerate(zip(methods_comp, colors_comp)):
    ratios = [r["comp_results"].get(method,{}).get("ratio",0)
              for r in all_results]
    ax.bar(x + i*width, ratios, width, label=method,
           color=color, alpha=0.85)
ax.axhline(y=4.0, color="black", ls="--", lw=1.5, label="4x target")
ax.set_title("Compression -- Ratio (all methods, all files)",
             fontweight="bold")
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(files_short, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Compression Ratio")
ax.legend(fontsize=7, ncol=3)
ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#ffffff")

# Panel D: Wavelet RMSE per file
ax = fig.add_subplot(gs[1, 2])
wav_rmse_list = [r["comp_results"].get("Wavelet 10%",{}).get("rmse",0)
                 for r in all_results]
bars = ax.bar(x, wav_rmse_list, color="#2ecc71", alpha=0.85, width=0.6)
for bar, val in zip(bars, wav_rmse_list):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8)
ax.set_title("Wavelet 10% -- RMSE per file\n(primary compression method)",
             fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(files_short, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("RMSE"); ax.grid(True, alpha=0.3, axis="y")
ax.set_facecolor("#ffffff")

plt.suptitle(
    f"PODFAM All Files Summary -- {len(all_results)} files processed\n"
    f"Classical Methods: 5 Denoising + Two-Colour Calibration + 4 Compression",
    fontsize=13, fontweight="bold", y=1.01)

plt.savefig("podfam_all_results.png", dpi=150,
            bbox_inches="tight", facecolor="#f8f9fa")
print("  Saved --> podfam_all_results.png")
plt.show()

print()
print("=" * 65)
print("ALL DONE!")
print("=" * 65)
print(f"  Files processed : {len(all_results)}")
print(f"  CSV saved       : podfam_all_results.csv")
print(f"  Plot saved      : podfam_all_results.png")
print(f"  Individual plots: one per file (*_result.png)")
print("=" * 65)
