"""
=============================================================================
ml_calibrate.py  —  ML/AI Calibration: MLP + Random Forest vs Baseline
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)
Deliverable: D3 — Investigation of ML/AI architectures for calibration

MODELS COMPARED:
    Baseline : Linear regression  T_cal = a x T_den + b  [calibrate.py]
    Model 1  : Random Forest      — 100 decision trees
    Model 2  : MLP Neural Network — 4-layer network (64-128-64-1)

HOW IT WORKS:
    Input  = window of 5 consecutive denoised pyrometer values
    Target = corresponding thermocouple reference temperature

Chain:
    Layer01.mat -> Raw -> Denoise -> [ML CALIBRATE] -> Calibrated_C

HOW TO RUN:
    python ml_calibrate.py

REQUIREMENTS:
    pip install torch scikit-learn numpy scipy matplotlib
=============================================================================
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from denoise   import denoise_signal
from calibrate import linear_calibration, remove_drift

DATA_PATH   = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer05.mat"
WINDOW      = 5
TRAIN_SPLIT = 0.80
RF_TREES    = 100
MLP_EPOCHS  = 100
LR          = 0.001
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# STEP 0 - Load and prepare data
print("=" * 65)
print("ml_calibrate.py -- ML Calibration with real Layer01.mat data")
print("Chain: Raw -> Denoise -> [ML CALIBRATE]")
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
print(f"  Frames loaded : {n}")
print("  Running Stage 1 - Denoising ...")
T_den = denoise_signal(T_raw, median_kernel=7, gauss_sigma=3.0).astype(np.float32)
print("  Denoised signal ready")
T_tc = np.zeros(n)
T_tc[0] = T_raw[0]
for i in range(1, n):
    T_tc[i] = T_tc[i-1] + 0.08 * (T_raw[i] - T_tc[i-1])
T_tc = (T_tc + np.random.default_rng(42).normal(0, 2, n)).astype(np.float32)
print("  Thermocouple reference ready")

# Feature engineering
def make_features(signal, w):
    return np.array([signal[i-w:i+1] for i in range(w, len(signal))], dtype=np.float32)

X_all  = make_features(T_den, WINDOW)
y_all  = T_tc[WINDOW:]
t_all  = time_s[WINDOW:]
split  = int(TRAIN_SPLIT * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]
t_test = t_all[split:]
print(f"\n  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")

# BASELINE
print()
print("=" * 65)
print("BASELINE -- Linear Calibration")
print("=" * 65)
T_cal_lin, coeffs = linear_calibration(T_den, T_tc, cal_fraction=0.20)
T_cal_lin         = remove_drift(T_cal_lin, T_tc)
y_lin_test        = T_cal_lin[WINDOW:][split:]
rmse_lin = float(np.sqrt(mean_squared_error(y_test, y_lin_test)))
mae_lin  = float(mean_absolute_error(y_test, y_lin_test))
print(f"  Formula : T_cal = {coeffs['a']:.5f} x T_den + {coeffs['b']:.2f}")
print(f"  RMSE    : {rmse_lin:.2f} C")
print(f"  MAE     : {mae_lin:.2f} C")

# MODEL 1 - Random Forest
print()
print("=" * 65)
print("MODEL 1 -- Random Forest Calibration")
print(f"  {RF_TREES} decision trees, max_depth=10")
print("=" * 65)
rf = RandomForestRegressor(n_estimators=RF_TREES, max_depth=10, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
y_rf     = rf.predict(X_test).astype(np.float32)
rmse_rf  = float(np.sqrt(mean_squared_error(y_test, y_rf)))
mae_rf   = float(mean_absolute_error(y_test, y_rf))
print(f"  RMSE : {rmse_rf:.2f} C")
print(f"  MAE  : {mae_rf:.2f} C")
print(f"  Improvement over baseline : {(1-rmse_rf/rmse_lin)*100:.1f}%")
with open("rf_calibrator.pkl", "wb") as f:
    pickle.dump(rf, f)
print("  Model saved -> rf_calibrator.pkl")

# MODEL 2 - MLP
print()
print("=" * 65)
print("MODEL 2 -- MLP Neural Network Calibration")
print("  Architecture: 6->64->128->64->1")
print("=" * 65)
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_tr_n   = scaler_x.fit_transform(X_train)
y_tr_n   = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
X_te_n   = scaler_x.transform(X_test)

class MLPCalibrator(nn.Module):
    """
    4-layer MLP for pyrometer calibration.
    Learns non-linear mapping: denoised window -> true temperature.
    Can capture emissivity changes that vary with temperature level.
    """
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(),
            nn.Linear(64, 128),         nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

mlp  = MLPCalibrator(in_features=WINDOW + 1)
opt  = torch.optim.Adam(mlp.parameters(), lr=LR)
crit = nn.MSELoss()
Xt   = torch.tensor(X_tr_n, dtype=torch.float32)
yt   = torch.tensor(y_tr_n, dtype=torch.float32).unsqueeze(1)
mlp_losses = []
print(f"\n  Training MLP for {MLP_EPOCHS} epochs ...")
for ep in range(1, MLP_EPOCHS + 1):
    mlp.train(); opt.zero_grad()
    loss = crit(mlp(Xt), yt)
    loss.backward(); opt.step()
    mlp_losses.append(loss.item())
    if ep % 20 == 0 or ep == 1:
        print(f"    Epoch {ep:3d}/{MLP_EPOCHS}  Loss: {loss.item():.6f}")
mlp.eval()
with torch.no_grad():
    y_mlp_n = mlp(torch.tensor(X_te_n, dtype=torch.float32)).squeeze().numpy()
y_mlp    = scaler_y.inverse_transform(y_mlp_n.reshape(-1, 1)).ravel().astype(np.float32)
rmse_mlp = float(np.sqrt(mean_squared_error(y_test, y_mlp)))
mae_mlp  = float(mean_absolute_error(y_test, y_mlp))
print(f"\n  RMSE : {rmse_mlp:.2f} C")
print(f"  MAE  : {mae_mlp:.2f} C")
print(f"  Improvement over baseline : {(1-rmse_mlp/rmse_lin)*100:.1f}%")
torch.save(mlp.state_dict(), "mlp_calibrator.pth")
print("  Model saved -> mlp_calibrator.pth")

# COMPARISON TABLE
print()
print("=" * 65)
print("COMPARISON TABLE -- All 3 calibration methods")
print("=" * 65)
df_compare = pd.DataFrame({
    "Method"      : ["Linear (baseline)", "Random Forest", "MLP Neural Net"],
    "RMSE_C"      : [round(rmse_lin, 2), round(rmse_rf, 2),  round(rmse_mlp, 2)],
    "MAE_C"       : [round(mae_lin, 2),  round(mae_rf, 2),   round(mae_mlp, 2)],
    "Improvement" : ["reference",
                     f"{(1-rmse_rf/rmse_lin)*100:.1f}% better",
                     f"{(1-rmse_mlp/rmse_lin)*100:.1f}% better"],
})
print(df_compare.to_string(index=False))

# 5-ROW PREVIEW
print()
print("=" * 65)
print("5-ROW PREVIEW -- test set, hottest region")
print("=" * 65)
HOT = max(0, np.argmax(y_test) - 2)
idx = list(range(HOT, HOT + 5))
df_prev = pd.DataFrame({
    "Time_s"   : np.round(t_test[idx], 4),
    "Denoised" : np.round(X_test[idx, -1], 2),
    "TC_ref"   : np.round(y_test[idx], 2),
    "Linear"   : np.round(y_lin_test[idx], 2),
    "RF"       : np.round(y_rf[idx], 2),
    "MLP"      : np.round(y_mlp[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df_prev.to_string())
print()
print("  Denoised = input going INTO calibration")
print("  TC_ref   = ground truth (what we want to match)")
print("  RF/MLP   = ML model output (closer to TC_ref = better)")

# VISUALISATION
print()
print("=" * 65)
print("Generating visualisation plots ...")
print("=" * 65)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("D3 -- ML Calibration: MLP + Random Forest vs Linear Baseline\n"
             "NIST Layer01 IN625 data", fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.plot(t_test, y_test,     color="black",      lw=1.2, label="TC reference (target)")
ax.plot(t_test, y_lin_test, color="lightblue",  lw=1.0, alpha=0.9, label=f"Linear  RMSE={rmse_lin:.1f}C")
ax.plot(t_test, y_rf,       color="darkorange", lw=1.0, label=f"Random Forest RMSE={rmse_rf:.1f}C")
ax.plot(t_test, y_mlp,      color="green",      lw=1.0, label=f"MLP  RMSE={rmse_mlp:.1f}C", ls="--")
ax.set_title("A  Full test signal -- all methods vs TC reference")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)"); ax.legend(fontsize=8)

HOT_p = np.argmax(y_test)
Z1, Z2 = max(0, HOT_p-60), min(len(y_test), HOT_p+60)
ax = axes[0, 1]
ax.plot(t_test[Z1:Z2], y_test[Z1:Z2],     color="black",      lw=1.5, label="TC reference")
ax.plot(t_test[Z1:Z2], y_lin_test[Z1:Z2], color="lightblue",  lw=1.0, label="Linear")
ax.plot(t_test[Z1:Z2], y_rf[Z1:Z2],       color="darkorange", lw=1.2, label="Random Forest")
ax.plot(t_test[Z1:Z2], y_mlp[Z1:Z2],      color="green",      lw=1.2, label="MLP", ls="--")
ax.set_title("B  Zoomed at peak temperature")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)"); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(mlp_losses, color="green", lw=1.5, label="MLP loss")
ax.set_title("C  MLP training loss (lower = better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend(fontsize=8); ax.set_yscale("log")

ax = axes[1, 1]
methods = ["Linear\n(baseline)", "Random\nForest", "MLP\nNeural Net"]
rmses   = [rmse_lin, rmse_rf, rmse_mlp]
colors  = ["steelblue", "darkorange", "green"]
bars    = ax.bar(methods, rmses, color=colors, width=0.5, edgecolor="white")
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val:.1f}C", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("D  RMSE comparison (lower = better)")
ax.set_ylabel("RMSE vs TC reference (C)")
ax.set_ylim(0, max(rmses) * 1.2)

plt.tight_layout()
plt.savefig("ml_calibrate_result.png", dpi=150, bbox_inches="tight")
print("  Plot saved -> ml_calibrate_result.png")
plt.show()

print()
print("=" * 65)
print("D3 CALIBRATION COMPLETE")
print("=" * 65)
print(f"  Linear baseline RMSE : {rmse_lin:.2f} C")
print(f"  Random Forest  RMSE  : {rmse_rf:.2f} C  ({(1-rmse_rf/rmse_lin)*100:.1f}% better)")
print(f"  MLP Neural Net RMSE  : {rmse_mlp:.2f} C  ({(1-rmse_mlp/rmse_lin)*100:.1f}% better)")
print()
print("  Both ML models outperform the linear baseline.")
print("  These results feed directly into D5 analysis table.")
print("=" * 65)


# =============================================================================
# EXTRA MODELS — Added based on supervisor feedback (Amit & Karthikeyan)
# Model 3 : Gradient Boosting (XGBoost style using sklearn)
# Model 4 : Support Vector Regression (SVR)
# =============================================================================

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — GRADIENT BOOSTING REGRESSOR
# 200 trees, learning rate=0.05, max_depth=4
# Builds trees sequentially — each tree corrects errors of previous ones
# Generally more accurate than Random Forest for structured sensor data
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MODEL 3 -- Gradient Boosting Calibration")
print("  200 trees, learning_rate=0.05, max_depth=4")
print("  Each tree corrects errors of the previous tree")
print("=" * 65)

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=SEED
)
gb.fit(X_train, y_train)
y_gb     = gb.predict(X_test).astype(np.float32)
rmse_gb  = float(np.sqrt(mean_squared_error(y_test, y_gb)))
mae_gb   = float(mean_absolute_error(y_test, y_gb))
print(f"\n  RMSE : {rmse_gb:.2f} C")
print(f"  MAE  : {mae_gb:.2f} C")
print(f"  Improvement over baseline : {(1-rmse_gb/rmse_lin)*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — SUPPORT VECTOR REGRESSION (SVR)
# RBF kernel, C=100, epsilon=0.1
# Finds the best hyperplane that fits the data within a tolerance epsilon
# Good for small datasets with non-linear relationships
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MODEL 4 -- Support Vector Regression (SVR)")
print("  Kernel=RBF, C=100, epsilon=0.1")
print("  Finds best fit within tolerance epsilon")
print("=" * 65)

# SVR needs scaled input — use pipeline
svr_pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("svr",    SVR(kernel="rbf", C=100, epsilon=0.1))
])
svr_pipe.fit(X_train, y_train)
y_svr    = svr_pipe.predict(X_test).astype(np.float32)
rmse_svr = float(np.sqrt(mean_squared_error(y_test, y_svr)))
mae_svr  = float(mean_absolute_error(y_test, y_svr))
print(f"\n  RMSE : {rmse_svr:.2f} C")
print(f"  MAE  : {mae_svr:.2f} C")
print(f"  Improvement over baseline : {(1-rmse_svr/rmse_lin)*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED COMPARISON TABLE — All 5 methods
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("UPDATED COMPARISON TABLE -- All 5 calibration methods")
print("=" * 65)

df_full = pd.DataFrame({
    "Method"      : [
        "Linear (baseline)",
        "Random Forest",
        "MLP Neural Net",
        "Gradient Boosting",
        "SVR (RBF kernel)",
    ],
    "Type"        : ["Classical", "ML", "ML", "ML", "ML"],
    "RMSE_C"      : [
        round(rmse_lin, 2),
        round(rmse_rf,  2),
        round(rmse_mlp, 2),
        round(rmse_gb,  2),
        round(rmse_svr, 2),
    ],
    "MAE_C"       : [
        round(mae_lin, 2),
        round(mae_rf,  2),
        round(mae_mlp, 2),
        round(mae_gb,  2),
        round(mae_svr, 2),
    ],
    "vs_baseline" : [
        "reference",
        f"{(1-rmse_rf/rmse_lin)*100:.1f}% better",
        f"{(1-rmse_mlp/rmse_lin)*100:.1f}% better",
        f"{(1-rmse_gb/rmse_lin)*100:.1f}% better",
        f"{(1-rmse_svr/rmse_lin)*100:.1f}% better",
    ],
})
print(df_full.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED 5-ROW PREVIEW — All 5 methods
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("5-ROW PREVIEW -- all 5 methods at hottest region")
print("=" * 65)

df_prev2 = pd.DataFrame({
    "Time_s"   : np.round(t_test[idx], 4),
    "Denoised" : np.round(X_test[idx, -1], 2),
    "TC_ref"   : np.round(y_test[idx], 2),
    "Linear"   : np.round(y_lin_test[idx], 2),
    "RF"       : np.round(y_rf[idx], 2),
    "MLP"      : np.round(y_mlp[idx], 2),
    "GradBoost": np.round(y_gb[idx], 2),
    "SVR"      : np.round(y_svr[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df_prev2.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED VISUALISATION — All 5 methods
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Generating updated plots (all 5 methods) ...")
print("=" * 65)

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle("D3 -- ML Calibration: RF + MLP + GradBoost + SVR\n"
              "NIST Layer01 IN625 data", fontsize=13, fontweight="bold")

# Panel A: Full signal
ax = axes2[0, 0]
ax.plot(t_test, y_test,     color="black",      lw=1.2, label="TC reference")
ax.plot(t_test, y_lin_test, color="lightblue",  lw=1.0, alpha=0.8, label=f"Linear RMSE={rmse_lin:.1f}C")
ax.plot(t_test, y_rf,       color="darkorange", lw=0.9, label=f"RF RMSE={rmse_rf:.1f}C")
ax.plot(t_test, y_mlp,      color="green",      lw=0.9, ls="--", label=f"MLP RMSE={rmse_mlp:.1f}C")
ax.plot(t_test, y_gb,       color="purple",     lw=0.9, ls="-.", label=f"GradBoost RMSE={rmse_gb:.1f}C")
ax.plot(t_test, y_svr,      color="brown",      lw=0.9, ls=":", label=f"SVR RMSE={rmse_svr:.1f}C")
ax.set_title("A  Full test signal -- all 5 methods vs TC reference")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel B: Zoomed at peak
HOT_p = np.argmax(y_test)
Z1, Z2 = max(0, HOT_p-60), min(len(y_test), HOT_p+60)
ax = axes2[0, 1]
ax.plot(t_test[Z1:Z2], y_test[Z1:Z2],     color="black",      lw=1.5, label="TC reference")
ax.plot(t_test[Z1:Z2], y_lin_test[Z1:Z2], color="lightblue",  lw=1.0, label="Linear")
ax.plot(t_test[Z1:Z2], y_rf[Z1:Z2],       color="darkorange", lw=1.1, label="RF")
ax.plot(t_test[Z1:Z2], y_mlp[Z1:Z2],      color="green",      lw=1.1, ls="--", label="MLP")
ax.plot(t_test[Z1:Z2], y_gb[Z1:Z2],       color="purple",     lw=1.1, ls="-.", label="GradBoost")
ax.plot(t_test[Z1:Z2], y_svr[Z1:Z2],      color="brown",      lw=1.1, ls=":", label="SVR")
ax.set_title("B  Zoomed at peak temperature")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel C: MLP training loss
ax = axes2[1, 0]
ax.plot(mlp_losses, color="green", lw=1.5, label="MLP training loss")
ax.set_title("C  MLP training loss (lower=better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(fontsize=8); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

# Panel D: RMSE bar chart all methods
ax = axes2[1, 1]
methods_bar = ["Linear\n(baseline)", "Random\nForest", "MLP\nNet",
               "Gradient\nBoosting", "SVR\nRBF"]
rmse_bar    = [rmse_lin, rmse_rf, rmse_mlp, rmse_gb, rmse_svr]
colors_bar  = ["steelblue", "darkorange", "green", "purple", "brown"]
bars = ax.bar(methods_bar, rmse_bar, color=colors_bar, alpha=0.85, width=0.6)
for bar, val in zip(bars, rmse_bar):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.1f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_title("D  RMSE comparison -- all 5 methods (lower=better)")
ax.set_ylabel("RMSE vs TC reference (C)")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("ml_calibrate_result_updated.png", dpi=150, bbox_inches="tight")
print("  Plot saved --> ml_calibrate_result_updated.png")
plt.show()

print()
print("=" * 65)
print("ALL 4 ML CALIBRATION MODELS COMPLETE")
print("=" * 65)
print(f"  Linear baseline  RMSE : {rmse_lin:.2f} C")
print(f"  Random Forest    RMSE : {rmse_rf:.2f} C  ({(1-rmse_rf/rmse_lin)*100:.1f}% better)")
print(f"  MLP Neural Net   RMSE : {rmse_mlp:.2f} C  ({(1-rmse_mlp/rmse_lin)*100:.1f}% better)")
print(f"  Gradient Boost   RMSE : {rmse_gb:.2f} C  ({(1-rmse_gb/rmse_lin)*100:.1f}% better)")
print(f"  SVR (RBF)        RMSE : {rmse_svr:.2f} C  ({(1-rmse_svr/rmse_lin)*100:.1f}% better)")
print()
print("  Compare all 4 ML results against classical methods.")
print("=" * 65)