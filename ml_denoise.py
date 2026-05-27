"""
=============================================================================
ml_denoise.py  —  ML/AI Denoising: CNN + LSTM vs Baseline
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)
Deliverable: D3 — Investigation of ML/AI architectures for denoising

MODELS COMPARED:
    Baseline : Median filter (k=7) + Gaussian smooth (σ=3)  [from denoise.py]
    Model 1  : 1-D CNN  — learns local noise patterns via convolution
    Model 2  : LSTM     — learns noise patterns over time sequences

HOW IT WORKS:
    Both models are trained as denoisers:
        Input  = noisy signal window
        Target = smooth (baseline denoised) signal
    After training, CNN/LSTM predictions are compared to the baseline.

HOW TO RUN:
    python ml_denoise.py

REQUIREMENTS:
    pip install torch numpy scipy matplotlib scikit-learn
=============================================================================
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from denoise import denoise_signal

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"
WINDOW_SIZE = 32       # input window length for CNN and LSTM
EPOCHS_CNN  = 50       # training epochs for CNN
EPOCHS_LSTM = 50       # training epochs for LSTM
BATCH_SIZE  = 32
LR          = 0.001    # learning rate
TRAIN_SPLIT = 0.80     # 80% train, 20% test
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")   # use CPU (GPU if available)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 0 — Loading Layer01.mat and preparing data")
print("=" * 65)

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

# Baseline denoised signal = training TARGET for both models
T_baseline = denoise_signal(T_raw, median_kernel=7, gauss_sigma=3.0
                            ).astype(np.float32)

print(f"  Signal length : {n} frames")
print(f"  Raw range     : {T_raw.min():.1f} – {T_raw.max():.1f} °C")
print(f"  Window size   : {WINDOW_SIZE}")

# Normalise to [0, 1] for stable training
scaler   = MinMaxScaler()
T_raw_n  = scaler.fit_transform(T_raw.reshape(-1, 1)).ravel().astype(np.float32)
T_base_n = scaler.transform(T_baseline.reshape(-1, 1)).ravel().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — sliding window pairs (noisy input, clean target)
# ─────────────────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """
    Creates overlapping windows of length WINDOW_SIZE.
    Input  = noisy raw window
    Target = corresponding baseline-denoised window
    """
    def __init__(self, noisy, clean, window):
        self.noisy  = noisy
        self.clean  = clean
        self.window = window

    def __len__(self):
        return len(self.noisy) - self.window

    def __getitem__(self, i):
        x = torch.tensor(self.noisy[i : i + self.window]).unsqueeze(0)  # (1, W)
        y = torch.tensor(self.clean[i : i + self.window]).unsqueeze(0)  # (1, W)
        return x, y


split     = int(TRAIN_SPLIT * n)
train_ds  = WindowDataset(T_raw_n[:split],  T_base_n[:split],  WINDOW_SIZE)
test_ds   = WindowDataset(T_raw_n[split:],  T_base_n[split:],  WINDOW_SIZE)
train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl   = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train windows : {len(train_ds)}")
print(f"  Test  windows : {len(test_ds)}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — 1-D CNN DENOISER
# Architecture:
#   Conv1d(1→16, k=7) → ReLU → Conv1d(16→32, k=5) → ReLU
#   → Conv1d(32→16, k=5) → ReLU → Conv1d(16→1, k=7)
#   All same-padding so output = input length
# ─────────────────────────────────────────────────────────────────────────────
class CNNDenoiser(nn.Module):
    """
    1-D Convolutional denoiser.
    Learns local noise patterns using stacked convolution layers.
    Each Conv1d layer scans a local window and learns to smooth it.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),   # local 7-step window
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),  # wider feature map
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),  # compress back
            nn.ReLU(),
            nn.Conv1d(16,  1, kernel_size=7, padding=3),  # output = clean signal
        )

    def forward(self, x):
        return self.net(x)   # x shape: (batch, 1, window)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — LSTM DENOISER
# Architecture:
#   LSTM(input=1, hidden=64, layers=2) → Linear(64→1)
#   The LSTM processes each time step and learns temporal patterns
# ─────────────────────────────────────────────────────────────────────────────
class LSTMDenoiser(nn.Module):
    """
    LSTM-based denoiser.
    Processes the signal step-by-step and learns to predict
    the clean value at each time step from its noisy context.
    """
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden,
                              num_layers=layers, batch_first=True)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        # x shape: (batch, 1, window) → swap to (batch, window, 1) for LSTM
        x   = x.squeeze(1).unsqueeze(2)      # (batch, window, 1)
        out, _ = self.lstm(x)                 # (batch, window, hidden)
        out = self.linear(out)                # (batch, window, 1)
        return out.squeeze(2).unsqueeze(1)    # (batch, 1, window)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_dl, epochs, lr, label):
    """
    Train a denoising model using MSE loss and Adam optimiser.
    Prints loss every 10 epochs.
    """
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses    = []

    print(f"\n  Training {label} for {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimiser.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_dl)
        losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  Loss: {avg_loss:.6f}")

    return losses


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, noisy_norm, window):
    """
    Run inference on the full signal using a sliding window.
    Averages overlapping predictions for a smooth output.
    """
    model.eval()
    predictions = np.zeros(len(noisy_norm), dtype=np.float32)
    counts      = np.zeros(len(noisy_norm), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(noisy_norm) - window, window // 2):
            x   = torch.tensor(noisy_norm[i:i+window]).unsqueeze(0).unsqueeze(0)
            out = model(x.to(DEVICE)).squeeze().cpu().numpy()
            predictions[i:i+window] += out
            counts[i:i+window]      += 1

    counts = np.maximum(counts, 1)
    pred_norm = predictions / counts

    # Inverse-transform back to °C
    pred_C = scaler.inverse_transform(pred_norm.reshape(-1, 1)).ravel()
    return pred_C.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(T_pred, T_target, label):
    """Calculate RMSE and MAE vs baseline target."""
    rmse = float(np.sqrt(np.mean((T_pred - T_target) ** 2)))
    mae  = float(np.mean(np.abs(T_pred - T_target)))
    print(f"  [{label}] RMSE vs baseline: {rmse:.2f} °C")
    print(f"  [{label}] MAE  vs baseline: {mae:.2f} °C")
    return rmse, mae


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — TRAIN CNN
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 1 — Training CNN Denoiser")
print("=" * 65)

cnn_model  = CNNDenoiser().to(DEVICE)
cnn_losses = train_model(cnn_model, train_dl, EPOCHS_CNN, LR, "CNN")
T_cnn      = predict(cnn_model, T_raw_n, WINDOW_SIZE)

print("\n  CNN evaluation:")
rmse_cnn, mae_cnn = evaluate(T_cnn, T_baseline, "CNN")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN LSTM
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 2 — Training LSTM Denoiser")
print("=" * 65)

lstm_model  = LSTMDenoiser(hidden=64, layers=2).to(DEVICE)
lstm_losses = train_model(lstm_model, train_dl, EPOCHS_LSTM, LR, "LSTM")
T_lstm      = predict(lstm_model, T_raw_n, WINDOW_SIZE)

print("\n  LSTM evaluation:")
rmse_lstm, mae_lstm = evaluate(T_lstm, T_baseline, "LSTM")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 3 — Comparison Table (all 3 methods)")
print("=" * 65)

rmse_base = float(np.sqrt(np.mean((T_baseline - T_baseline) ** 2)))
mae_base  = float(np.mean(np.abs(T_raw - T_baseline)))

df_compare = pd.DataFrame({
    "Method"       : ["Baseline (median+Gaussian)", "CNN Denoiser", "LSTM Denoiser"],
    "RMSE_vs_base" : [0.0,       round(rmse_cnn,  2), round(rmse_lstm, 2)],
    "MAE_vs_base"  : [0.0,       round(mae_cnn,   2), round(mae_lstm,  2)],
    "Description"  : [
        "Traditional: median filter + Gaussian smooth",
        "ML: 1-D CNN learns local noise patterns",
        "ML: LSTM learns temporal noise sequences",
    ]
})
print(df_compare.to_string(index=False))
print()
print("  Note: RMSE/MAE measured against baseline denoised signal.")
print("  Lower = closer to baseline. CNN/LSTM learn to mimic the baseline.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — 5-ROW PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 4 — 5-Row Preview at hottest region")
print("=" * 65)

HOT = max(0, T_raw.argmax() - 2)
idx = list(range(HOT, HOT + 5))

df_prev = pd.DataFrame({
    "Time_s"    : np.round(time_s[idx], 4),
    "Raw_C"     : np.round(T_raw[idx], 2),
    "Baseline"  : np.round(T_baseline[idx], 2),
    "CNN"       : np.round(T_cnn[idx], 2),
    "LSTM"      : np.round(T_lstm[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df_prev.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — VISUALISATION (4 plots)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 5 — Generating visualisation plots ...")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("D3 — ML Denoising: CNN vs LSTM vs Baseline\n"
             "NIST Layer01 IN625 data", fontsize=13, fontweight="bold")

# ── Panel A: Full signal ─────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(time_s, T_raw,      color="lightcoral",  lw=0.6, alpha=0.5, label="Raw")
ax.plot(time_s, T_baseline, color="steelblue",   lw=1.2, label="Baseline (median+Gaussian)")
ax.plot(time_s, T_cnn,      color="darkorange",  lw=1.0, label="CNN denoiser")
ax.plot(time_s, T_lstm,     color="green",       lw=1.0, label="LSTM denoiser", ls="--")
ax.set_title("A  Full signal — all methods")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8)

# ── Panel B: Zoomed in at peak ───────────────────────────────────────────────
Z1, Z2 = max(0, HOT - 80), min(n, HOT + 80)
ax = axes[0, 1]
ax.plot(time_s[Z1:Z2], T_raw[Z1:Z2],      color="lightcoral", lw=0.8, alpha=0.6, label="Raw")
ax.plot(time_s[Z1:Z2], T_baseline[Z1:Z2], color="steelblue",  lw=1.5, label="Baseline")
ax.plot(time_s[Z1:Z2], T_cnn[Z1:Z2],      color="darkorange", lw=1.2, label="CNN")
ax.plot(time_s[Z1:Z2], T_lstm[Z1:Z2],     color="green",      lw=1.2, label="LSTM", ls="--")
ax.set_title("B  Zoomed in at peak temperature")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=8)

# ── Panel C: Training loss curves ───────────────────────────────────────────
ax = axes[1, 0]
ax.plot(cnn_losses,  color="darkorange", lw=1.5, label="CNN loss")
ax.plot(lstm_losses, color="green",      lw=1.5, label="LSTM loss", ls="--")
ax.set_title("C  Training loss (lower = better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(fontsize=8)
ax.set_yscale("log")

# ── Panel D: Error vs baseline ───────────────────────────────────────────────
ax = axes[1, 1]
err_cnn  = np.abs(T_cnn  - T_baseline)
err_lstm = np.abs(T_lstm - T_baseline)
ax.plot(time_s, err_cnn,  color="darkorange", lw=0.8, alpha=0.7, label=f"CNN  MAE={mae_cnn:.1f}°C")
ax.plot(time_s, err_lstm, color="green",      lw=0.8, alpha=0.7, label=f"LSTM MAE={mae_lstm:.1f}°C")
ax.axhline(y=np.mean(err_cnn),  color="darkorange", ls="--", lw=1.0)
ax.axhline(y=np.mean(err_lstm), color="green",      ls="--", lw=1.0)
ax.set_title("D  Absolute error vs baseline")
ax.set_xlabel("Time (s)"); ax.set_ylabel("|Error| (°C)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("ml_denoise_result.png", dpi=150, bbox_inches="tight")
print("  Plot saved → ml_denoise_result.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────
torch.save(cnn_model.state_dict(),  "cnn_denoiser.pth")
torch.save(lstm_model.state_dict(), "lstm_denoiser.pth")
print("  Models saved → cnn_denoiser.pth, lstm_denoiser.pth")

print()
print("=" * 65)
print("D3 DENOISING COMPLETE")
print("=" * 65)
print(f"  Baseline (median+Gaussian) : reference")
print(f"  CNN  RMSE vs baseline      : {rmse_cnn:.2f} °C")
print(f"  LSTM RMSE vs baseline      : {rmse_lstm:.2f} °C")
print()
print("  Lower RMSE = model learned to denoise like the baseline.")
print("  These results feed directly into D5 analysis table.")
print("=" * 65)


# =============================================================================
# EXTRA MODELS — Added based on supervisor feedback (Amit & Karthikeyan)
# Model 3 : Autoencoder Denoiser
# Model 4 : Bidirectional LSTM (BiLSTM)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — AUTOENCODER DENOISER
# Architecture:
#   Encoder: Linear(32→16) → ReLU → Linear(16→8)  [bottleneck]
#   Decoder: Linear(8→16)  → ReLU → Linear(16→32)
#   Learns compressed representation then reconstructs clean signal
# ─────────────────────────────────────────────────────────────────────────────
class AutoencoderDenoiser(nn.Module):
    """
    Autoencoder-based denoiser.
    Encoder compresses noisy window into bottleneck (8 values).
    Decoder reconstructs clean signal from compressed representation.
    Bottleneck forces the model to discard noise and keep only
    the important signal structure.
    """
    def __init__(self, window=32, bottleneck=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window, 16),
            nn.ReLU(),
            nn.Linear(16, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16),
            nn.ReLU(),
            nn.Linear(16, window),
        )

    def forward(self, x):
        # x shape: (batch, 1, window)
        x_flat = x.squeeze(1)           # (batch, window)
        z      = self.encoder(x_flat)   # (batch, bottleneck)
        out    = self.decoder(z)        # (batch, window)
        return out.unsqueeze(1)         # (batch, 1, window)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — BIDIRECTIONAL LSTM (BiLSTM) DENOISER
# Architecture:
#   BiLSTM(input=1, hidden=64, layers=2, bidirectional=True)
#   → Linear(128→1)
#   Processes signal in BOTH forward and backward directions
#   Better than standard LSTM for denoising because it uses
#   both past and future context for each prediction
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMDenoiser(nn.Module):
    """
    Bidirectional LSTM denoiser.
    Processes the signal in both forward and backward directions.
    Each time step prediction uses both past AND future context.
    Better than standard LSTM for denoising — not just for real-time.
    hidden*2 because BiLSTM concatenates forward + backward hidden states.
    """
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm   = nn.LSTM(input_size=1, hidden_size=hidden,
                              num_layers=layers, batch_first=True,
                              bidirectional=True)
        self.linear = nn.Linear(hidden * 2, 1)  # *2 for bidirectional

    def forward(self, x):
        # x shape: (batch, 1, window) → (batch, window, 1) for LSTM
        x       = x.squeeze(1).unsqueeze(2)      # (batch, window, 1)
        out, _  = self.lstm(x)                   # (batch, window, hidden*2)
        out     = self.linear(out)               # (batch, window, 1)
        return out.squeeze(2).unsqueeze(1)       # (batch, 1, window)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN AUTOENCODER DENOISER
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 6 — Training Autoencoder Denoiser (Model 3)")
print("=" * 65)

EPOCHS_AE = 50

ae_model   = AutoencoderDenoiser(window=WINDOW_SIZE, bottleneck=8).to(DEVICE)
ae_losses  = train_model(ae_model, train_dl, EPOCHS_AE, LR, "Autoencoder")
T_ae       = predict(ae_model, T_raw_n, WINDOW_SIZE)

print("\n  Autoencoder evaluation:")
rmse_ae, mae_ae = evaluate(T_ae, T_baseline, "Autoencoder")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — TRAIN BIDIRECTIONAL LSTM
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 7 — Training Bidirectional LSTM Denoiser (Model 4)")
print("=" * 65)

EPOCHS_BILSTM = 50

bilstm_model  = BiLSTMDenoiser(hidden=64, layers=2).to(DEVICE)
bilstm_losses = train_model(bilstm_model, train_dl, EPOCHS_BILSTM, LR, "BiLSTM")
T_bilstm      = predict(bilstm_model, T_raw_n, WINDOW_SIZE)

print("\n  BiLSTM evaluation:")
rmse_bilstm, mae_bilstm = evaluate(T_bilstm, T_baseline, "BiLSTM")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — UPDATED COMPARISON TABLE (all 5 methods)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 8 — Full Comparison Table (all 5 methods)")
print("=" * 65)

df_full = pd.DataFrame({
    "Method"       : [
        "Baseline (median+Gaussian)",
        "CNN Denoiser",
        "LSTM Denoiser",
        "Autoencoder Denoiser",
        "BiLSTM Denoiser",
    ],
    "Type"         : ["Classical", "ML", "ML", "ML", "ML"],
    "RMSE_vs_base" : [
        0.0,
        round(rmse_cnn,    2),
        round(rmse_lstm,   2),
        round(rmse_ae,     2),
        round(rmse_bilstm, 2),
    ],
    "MAE_vs_base"  : [
        0.0,
        round(mae_cnn,    2),
        round(mae_lstm,   2),
        round(mae_ae,     2),
        round(mae_bilstm, 2),
    ],
    "Description"  : [
        "Traditional: median filter + Gaussian smooth",
        "ML: 4 Conv1d layers, learns local noise patterns",
        "ML: 2-layer LSTM, learns temporal patterns",
        "ML: Encoder-Decoder with bottleneck=8",
        "ML: Bidirectional LSTM, uses past + future context",
    ]
})
print(df_full.to_string(index=False))
print()
print("  Note: Lower RMSE = closer to baseline denoised signal.")
print("  All ML models learn to mimic the median+Gaussian baseline.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — UPDATED 5-ROW PREVIEW (all 5 methods)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 9 — 5-Row Preview at hottest region (all 5 methods)")
print("=" * 65)

df_prev2 = pd.DataFrame({
    "Time_s"    : np.round(time_s[idx], 4),
    "Raw_C"     : np.round(T_raw[idx],     2),
    "Baseline"  : np.round(T_baseline[idx],2),
    "CNN"       : np.round(T_cnn[idx],     2),
    "LSTM"      : np.round(T_lstm[idx],    2),
    "Autoenc"   : np.round(T_ae[idx],      2),
    "BiLSTM"    : np.round(T_bilstm[idx],  2),
}, index=[f"t{i}" for i in idx])
print(df_prev2.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — UPDATED VISUALISATION (all 5 methods)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STEP 10 — Generating updated plots (all 5 methods)")
print("=" * 65)

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle("D3 — ML Denoising: CNN + LSTM + Autoencoder + BiLSTM\n"
              "NIST Layer01 IN625 data", fontsize=13, fontweight="bold")

# Panel A: Full signal
ax = axes2[0, 0]
ax.plot(time_s, T_raw,      color="lightcoral",  lw=0.5, alpha=0.4, label="Raw")
ax.plot(time_s, T_baseline, color="steelblue",   lw=1.2, label="Baseline")
ax.plot(time_s, T_cnn,      color="darkorange",  lw=0.9, label="CNN")
ax.plot(time_s, T_lstm,     color="green",       lw=0.9, ls="--", label="LSTM")
ax.plot(time_s, T_ae,       color="purple",      lw=0.9, ls="-.", label="Autoencoder")
ax.plot(time_s, T_bilstm,   color="brown",       lw=0.9, ls=":", label="BiLSTM")
ax.set_title("A  Full signal — all 5 methods")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel B: Zoomed at peak
Z1, Z2 = max(0, HOT - 80), min(n, HOT + 80)
ax = axes2[0, 1]
ax.plot(time_s[Z1:Z2], T_raw[Z1:Z2],      color="lightcoral", lw=0.8, alpha=0.5, label="Raw")
ax.plot(time_s[Z1:Z2], T_baseline[Z1:Z2], color="steelblue",  lw=1.4, label="Baseline")
ax.plot(time_s[Z1:Z2], T_cnn[Z1:Z2],      color="darkorange", lw=1.1, label="CNN")
ax.plot(time_s[Z1:Z2], T_lstm[Z1:Z2],     color="green",      lw=1.1, ls="--", label="LSTM")
ax.plot(time_s[Z1:Z2], T_ae[Z1:Z2],       color="purple",     lw=1.1, ls="-.", label="Autoencoder")
ax.plot(time_s[Z1:Z2], T_bilstm[Z1:Z2],   color="brown",      lw=1.1, ls=":", label="BiLSTM")
ax.set_title("B  Zoomed at peak temperature")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel C: Training loss curves
ax = axes2[1, 0]
ax.plot(cnn_losses,    color="darkorange", lw=1.4, label="CNN")
ax.plot(lstm_losses,   color="green",      lw=1.4, ls="--", label="LSTM")
ax.plot(ae_losses,     color="purple",     lw=1.4, ls="-.", label="Autoencoder")
ax.plot(bilstm_losses, color="brown",      lw=1.4, ls=":", label="BiLSTM")
ax.set_title("C  Training loss curves (lower=better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(fontsize=8); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

# Panel D: RMSE bar chart
ax = axes2[1, 1]
methods_bar = ["Baseline","CNN","LSTM","Autoencoder","BiLSTM"]
rmse_bar    = [0.0, rmse_cnn, rmse_lstm, rmse_ae, rmse_bilstm]
colors_bar  = ["steelblue","darkorange","green","purple","brown"]
bars = ax.bar(methods_bar, rmse_bar, color=colors_bar, alpha=0.85, width=0.6)
for bar, val in zip(bars, rmse_bar):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title("D  RMSE vs baseline (lower=better)")
ax.set_ylabel("RMSE (C)"); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("ml_denoise_result_updated.png", dpi=150, bbox_inches="tight")
print("  Plot saved --> ml_denoise_result_updated.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ALL MODELS
# ─────────────────────────────────────────────────────────────────────────────
torch.save(ae_model.state_dict(),     "autoencoder_denoiser.pth")
torch.save(bilstm_model.state_dict(), "bilstm_denoiser.pth")
print("  Models saved --> autoencoder_denoiser.pth, bilstm_denoiser.pth")

print()
print("=" * 65)
print("ALL 4 ML DENOISING MODELS COMPLETE")
print("=" * 65)
print(f"  Baseline (median+Gaussian) : reference")
print(f"  CNN          RMSE : {rmse_cnn:.2f} C")
print(f"  LSTM         RMSE : {rmse_lstm:.2f} C")
print(f"  Autoencoder  RMSE : {rmse_ae:.2f} C")
print(f"  BiLSTM       RMSE : {rmse_bilstm:.2f} C")
print()
print("  Compare all 4 ML results against classical methods")
print("  in classical_methods.py for full D3 comparison.")
print("=" * 65)