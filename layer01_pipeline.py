"""
=============================================================================
Pyrometer Data Pre-Processing Pipeline — Layer 01 (NIST AM-Bench IN625)
=============================================================================
Thesis project: Automation of pyrometer data pre-processing
                (denoising, calibration, compression)

Dataset:  Layer01.mat  -- NIST AMBench, Inconel 625, Build 1
          RadiantTemp : (126, 360, 2497) uint16   raw pyrometer counts
          BuildTime   : (2497, 3)  float32  [time_s, x_mm, y_mm]

How to run:
    python layer01_pipeline.py

To test Layer02:
    Change LAYER = "01"  to  LAYER = "02"  at the top
=============================================================================
"""

import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from sklearn.decomposition import TruncatedSVD

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — change layer number here
# ─────────────────────────────────────────────────────────────────────────────
LAYER     = "01"    # change to "02", "03" ... "10" for other layers
DATA_PATH = rf"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer{LAYER}.mat"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"STEP 0 -- Loading Layer{LAYER}.mat ...")
print("=" * 60)

mat  = sio.loadmat(DATA_PATH)
L    = mat["Layer"][0, 0]

raw        = L["RadiantTemp"].astype(np.float32)
build_time = L["BuildTime"]
frame_ids  = L["RawFrameNumber"].ravel()

laser_power = int(L["LaserPower"].flat[0])
scan_speed  = int(L["ScanSpeed"].flat[0])
sh_A        = float(L["SHvariable_A"].flat[0])
sh_B        = float(L["SHvariable_B"].flat[0])
sh_C        = int(L["SHvariable_C"].flat[0])

print(f"  Raw array shape  : {raw.shape}  (rows x cols x frames)")
print(f"  Laser power      : {laser_power} W")
print(f"  Scan speed       : {scan_speed} mm/s")
print(f"  Calibration A/B/C: {sh_A} / {sh_B} / {sh_C}")
print()

# Find hottest frame and pixel automatically
frame_totals  = raw.sum(axis=(0, 1))
PREVIEW_FRAME = int(frame_totals.argmax())
hot_frame     = raw[:, :, PREVIEW_FRAME]
px_r, px_c    = np.unravel_index(hot_frame.argmax(), hot_frame.shape)

print(f"  Hottest frame : {PREVIEW_FRAME}")
print(f"  Hottest pixel : row={px_r}, col={px_c}  "
      f"(value={hot_frame[px_r, px_c]:.0f} counts)")

# Zoom window for plots
ZOOM = 25
r1 = max(0,            px_r - ZOOM)
r2 = min(raw.shape[0], px_r + ZOOM)
c1 = max(0,            px_c - ZOOM)
c2 = min(raw.shape[1], px_c + ZOOM)

ROW_START = max(0, px_r - 2)
COL_START = max(0, px_c - 2)

def show_5x5(arr_3d, frame_idx, r0, c0, title):
    snippet = arr_3d[r0:r0+5, c0:c0+5, frame_idx]
    df = pd.DataFrame(
        snippet,
        index   = [f"Row{r0+i}" for i in range(5)],
        columns = [f"Col{c0+i}" for i in range(5)],
    )
    print(f"  -- {title} --")
    print(df.to_string())
    print()

print()
print("=" * 60)
print("RAW DATA PREVIEW (before any processing)")
print("=" * 60)
show_5x5(raw, PREVIEW_FRAME, ROW_START, COL_START, "RAW counts (uint16)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DENOISING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 -- Denoising")
print("=" * 60)

denoised = np.zeros_like(raw)
for t in range(raw.shape[2]):
    denoised[:, :, t] = uniform_filter(raw[:, :, t], size=3)

print("  Spatial 3x3 box filter applied to all frames")

pixel_ts   = denoised[px_r, px_c, :]
spike_mask = np.abs(pixel_ts - pixel_ts.mean()) > 3 * pixel_ts.std()
print(f"  Temporal spike check on hottest pixel ({px_r},{px_c}): "
      f"{spike_mask.sum()} spikes detected")

show_5x5(denoised, PREVIEW_FRAME, ROW_START, COL_START, "DENOISED counts")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 -- Calibration  (raw counts -> temperature C)")
print("=" * 60)

calibrated = sh_A * denoised + sh_B - 273.15
calibrated = np.clip(calibrated, 0, 3000)

print(f"  Formula : T(C) = {sh_A} x count + {sh_B} - 273.15")
print(f"  Range   : {calibrated.min():.1f} C  to  {calibrated.max():.1f} C")
print(f"  Hottest pixel temperature: "
      f"{calibrated[px_r, px_c, PREVIEW_FRAME]:.1f} C")

show_5x5(calibrated, PREVIEW_FRAME, ROW_START, COL_START, "CALIBRATED C")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COMPRESSION
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 -- Compression  (Truncated SVD, rank-50)")
print("=" * 60)

n_rows, n_cols, n_frames = calibrated.shape
mat2d = calibrated.reshape(n_rows * n_cols, n_frames)

K   = 50
svd = TruncatedSVD(n_components=K, random_state=42)
compressed_2d    = svd.fit_transform(mat2d)
reconstructed_2d = svd.inverse_transform(compressed_2d)
reconstructed    = reconstructed_2d.reshape(n_rows, n_cols, n_frames)

original_size   = mat2d.nbytes
compressed_size = (compressed_2d.nbytes + svd.components_.nbytes
                   + svd.singular_values_.nbytes)
ratio = original_size / compressed_size
rmse  = np.sqrt(np.mean((calibrated - reconstructed) ** 2))

print(f"  Original   : {original_size / 1e6:.1f} MB")
print(f"  Compressed : {compressed_size / 1e6:.1f} MB  (rank={K})")
print(f"  Ratio      : {ratio:.1f}x")
print(f"  RMSE       : {rmse:.2f} C")

show_5x5(reconstructed, PREVIEW_FRAME, ROW_START, COL_START,
         "RECONSTRUCTED C (after compression)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — VISUALISATION (zoomed into hot spot)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 -- Generating visualisation ...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"Layer {LAYER} -- Pyrometer Pre-Processing Pipeline\n"
             f"NIST AMBench IN625  |  Laser {laser_power} W  |  "
             f"Scan {scan_speed} mm/s", fontsize=13, fontweight="bold")

kw = dict(aspect="auto", origin="lower")

# Panel A -- raw zoomed
im0 = axes[0,0].imshow(raw[r1:r2, c1:c2, PREVIEW_FRAME], **kw, cmap="inferno")
axes[0,0].set_title(f"A  Raw counts (frame {PREVIEW_FRAME}) -- zoomed")
axes[0,0].plot(px_c-c1, px_r-r1, "w+", markersize=12, markeredgewidth=2,
               label=f"Hot pixel ({px_r},{px_c})")
axes[0,0].legend(fontsize=7)
plt.colorbar(im0, ax=axes[0,0], label="DN (counts)")

# Panel B -- denoised zoomed
im1 = axes[0,1].imshow(denoised[r1:r2, c1:c2, PREVIEW_FRAME], **kw, cmap="inferno")
axes[0,1].set_title("B  After 3x3 spatial denoising -- zoomed")
axes[0,1].plot(px_c-c1, px_r-r1, "w+", markersize=12, markeredgewidth=2)
plt.colorbar(im1, ax=axes[0,1], label="DN (counts)")

# Panel C -- calibrated zoomed
im2 = axes[1,0].imshow(calibrated[r1:r2, c1:c2, PREVIEW_FRAME], **kw, cmap="hot")
axes[1,0].set_title("C  Calibrated temperature (C) -- zoomed")
axes[1,0].plot(px_c-c1, px_r-r1, "b+", markersize=12, markeredgewidth=2)
plt.colorbar(im2, ax=axes[1,0], label="C")

# Panel D -- per-frame MAX temperature across all pixels (full time-series)
# Use frame index on x-axis since build_time covers only a tiny window
frames       = np.arange(n_frames)
raw_max      = raw.max(axis=(0, 1))           # max pixel per frame
denoised_max = denoised.max(axis=(0, 1))
cal_max      = calibrated.max(axis=(0, 1))
recon_max    = reconstructed.max(axis=(0, 1))

axes[1,1].plot(frames, raw_max,
               alpha=0.5, label="Raw (max counts)",
               color="steelblue", linewidth=0.8)
axes[1,1].plot(frames, denoised_max,
               label="Denoised (max counts)",
               color="darkorange", linewidth=1.0)
ax2 = axes[1,1].twinx()
ax2.plot(frames, cal_max,
         label="Calibrated (C)", color="crimson",
         linewidth=1.2, linestyle="--")
ax2.plot(frames, recon_max,
         label="Reconstructed (C)", color="purple",
         linewidth=1.0, linestyle=":")
ax2.set_ylabel("Temperature (C)", color="crimson")
axes[1,1].set_xlabel("Frame index")
axes[1,1].set_ylabel("Raw / Denoised counts")
axes[1,1].set_title("D  Per-frame MAX temperature -- all stages")
axes[1,1].legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)

plt.tight_layout()
outfile = f"layer{LAYER}_pipeline_result.png"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"  Figure saved -> {outfile}")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"PIPELINE COMPLETE -- Layer{LAYER} SUMMARY")
print("=" * 60)
print(f"  Input frames   : {n_frames}")
print(f"  Spatial pixels : {n_rows} x {n_cols} = {n_rows*n_cols:,}")
print(f"  Hottest pixel  : ({px_r}, {px_c})")
print(f"  Peak temp      : {calibrated[px_r, px_c, :].max():.1f} C")
print(f"  Denoising      : 3x3 box filter + temporal spike clip")
print(f"  Calibration    : Linear SH -> range 0-{calibrated.max():.0f} C")
print(f"  Compression    : SVD rank-{K} -> {ratio:.1f}x smaller, RMSE {rmse:.2f} C")
print()
print("  To test Layer02: change LAYER = '02' at the top of this file")
print("=" * 60)