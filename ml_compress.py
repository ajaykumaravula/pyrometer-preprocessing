"""
=============================================================================
ml_compress.py  --  ML/AI Compression: Autoencoder + PCA vs Baseline
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)
Deliverable: D3 -- Investigation of ML/AI architectures for compression

MODELS COMPARED:
    Baseline 1 : Wavelet compression (keep 10% coefficients) [compress.py]
    Baseline 2 : SVD rank-10 compression                     [compress.py]
    Model 1    : PCA (3 components)  -- ML dimensionality reduction
    Model 2    : Autoencoder         -- neural network compress + reconstruct

HOW IT WORKS:
    Signal is split into overlapping windows of 32 samples.
    Each window is compressed to 3 values (bottleneck).
    Then reconstructed back to 32 samples.
    Compression ratio = 32/3 = 10.7x

Chain:
    Layer01.mat -> Raw -> Denoise -> Calibrate -> [ML COMPRESS]

HOW TO RUN:
    python ml_compress.py

REQUIREMENTS:
    pip install torch scikit-learn numpy scipy matplotlib
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from denoise   import denoise_signal
from calibrate import linear_calibration, remove_drift
from compress  import wavelet_compress, wavelet_reconstruct, \
                      svd_compress, svd_reconstruct

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = "clean_pyrometer_data.csv"
WINDOW      = 32      # window size for sliding window compression
BOTTLENECK  = 3       # compressed dimension (3 values per 32 samples)
AE_EPOCHS   = 100     # autoencoder training epochs
LR          = 0.001
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 -- Load and run full pipeline chain
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("ml_compress.py -- ML Compression with local CSV data")
print("Chain: Raw -> Denoise -> Calibrate -> [ML COMPRESS]")
print("=" * 65)

print(f"\n  Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
T_raw = df["pyr1_raw_C"].values.astype(np.float32)
T_tc_orig = df["tc_ref_C"].values.astype(np.float32)

mask  = T_raw > 10
T_raw = T_raw[mask]
T_tc  = T_tc_orig[mask]
n     = len(T_raw)
time_s = df["time_s"].values[mask]

print("  Stage 1 - Denoising ...")
T_den = denoise_signal(T_raw, median_kernel=7, gauss_sigma=3.0).astype(np.float32)

print("  Stage 2 - Calibrating ...")
T_cal, coeffs = linear_calibration(T_den, T_tc, cal_fraction=0.20)
T_cal         = remove_drift(T_cal, T_tc).astype(np.float32)
print(f"  Calibrated signal ready — range: {T_cal.min():.1f} - {T_cal.max():.1f} C")

# Show what goes INTO compression
HOT = max(0, T_cal.argmax() - 2)
idx = list(range(HOT, HOT + 5))
print()
print("=" * 65)
print("COMPRESS INPUT -- Calibrated_C (from calibrate.py)")
print("=" * 65)
df_in = pd.DataFrame({
    "Time_s"       : np.round(time_s[idx], 4),
    "Calibrated_C" : np.round(T_cal[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df_in.to_string())
print(f"  Original size : {T_cal.nbytes} bytes ({T_cal.nbytes/1e6:.4f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Build sliding window matrix
# ─────────────────────────────────────────────────────────────────────────────
def make_windows(sig, w):
    """Split signal into overlapping windows of length w."""
    return np.array([sig[i:i+w] for i in range(len(sig)-w)], dtype=np.float32)

def windows_to_signal(windows, orig_len):
    """Average overlapping windows back into a 1-D signal."""
    out = np.zeros(orig_len, dtype=np.float32)
    cts = np.zeros(orig_len, dtype=np.float32)
    for i, win in enumerate(windows):
        out[i:i+len(win)] += win
        cts[i:i+len(win)] += 1
    return out / np.maximum(cts, 1)

X_all  = make_windows(T_cal, WINDOW)   # (N, 32)
scaler = MinMaxScaler()
X_n    = scaler.fit_transform(X_all)   # normalised [0,1]
print(f"\n  Window matrix : {X_all.shape}  (windows x window_size)")


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 1 -- Wavelet compression
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("BASELINE 1 -- Wavelet Compression (keep 10%)")
print("=" * 65)
cw       = wavelet_compress(T_cal, keep_fraction=0.10)
T_wav    = wavelet_reconstruct(cw)
rmse_wav = float(np.sqrt(mean_squared_error(T_cal, T_wav)))
ratio_wav = T_cal.nbytes / max(1, cw["nonzero"] * 8)
print(f"  RMSE             : {rmse_wav:.2f} C")
print(f"  Compression ratio: {ratio_wav:.1f}x")
print(f"  Compressed size  : {cw['nonzero']*8} bytes")


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE 2 -- SVD compression
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("BASELINE 2 -- SVD Compression (rank=10)")
print("=" * 65)
sig_2d    = T_cal[:2048].reshape(64, 32).astype(np.float64)
cs        = svd_compress(sig_2d, rank=10)
rs        = svd_reconstruct(cs)
T_svd_raw = rs.ravel().astype(np.float32)
T_svd     = np.interp(np.arange(n),
                      np.linspace(0, n-1, len(T_svd_raw)),
                      T_svd_raw).astype(np.float32)
rmse_svd  = float(np.sqrt(mean_squared_error(T_cal, T_svd)))
ratio_svd = sig_2d.nbytes / (cs["U"].nbytes + cs["S"].nbytes + cs["Vt"].nbytes)
print(f"  RMSE             : {rmse_svd:.2f} C")
print(f"  Compression ratio: {ratio_svd:.1f}x")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 -- Delta Encoding (ATP-3 recommended near-lossless)
# ─────────────────────────────────────────────────────────────────────────────
from compress import delta_compress, delta_reconstruct
print()
print("=" * 65)
print("STAGE 3 -- Delta Encoding (Near-Lossless)")
print("=" * 65)
comp_delta = delta_compress(T_cal, scale=100)
T_delta = delta_reconstruct(comp_delta)
rmse_delta = float(np.sqrt(mean_squared_error(T_cal, T_delta)))
ratio_delta = (T_cal.nbytes) / (8 + len(comp_delta["diffs"]) * 2)
print(f"  Delta Encoding RMSE : {rmse_delta:.4f} C")
print(f"  Delta Encoding Ratio: {ratio_delta:.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 -- ML Compression (PCA + Autoencoder + VAE)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("STAGE 4 -- ML/AI Architectures (Autoencoder, VAE, PCA)")
print(f"  Target ratio: {WINDOW/BOTTLENECK:.1f}x")
print("=" * 65)

# PCA
pca = PCA(n_components=BOTTLENECK)
X_pca_n = pca.fit_transform(X_n)
X_pca_orig = scaler.inverse_transform(pca.inverse_transform(X_pca_n))
T_pca = windows_to_signal(X_pca_orig, n)
rmse_pca = float(np.sqrt(mean_squared_error(T_cal, T_pca)))
ratio_pca = WINDOW / BOTTLENECK

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim=32, bottleneck=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
            nn.Linear(8, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 8), nn.ReLU(),
            nn.Linear(8, 16),         nn.ReLU(),
            nn.Linear(16, input_dim),
        )
    def forward(self, x): return self.decoder(self.encoder(x))

# VAE
class VAE(nn.Module):
    def __init__(self, input_dim=32, bottleneck=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2_mu = nn.Linear(16, bottleneck)
        self.fc2_logvar = nn.Linear(16, bottleneck)
        self.fc3 = nn.Linear(bottleneck, 16)
        self.fc4 = nn.Linear(16, input_dim)
        self.relu = nn.ReLU()
    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = self.relu(self.fc3(z))
        return self.fc4(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

Xt = torch.tensor(X_n, dtype=torch.float32)

# Train Autoencoder
ae = Autoencoder(input_dim=WINDOW, bottleneck=BOTTLENECK)
opt_ae = torch.optim.Adam(ae.parameters(), lr=LR)
ae_losses = []
print(f"\n  Training Autoencoder ({AE_EPOCHS} epochs) ...")
for ep in range(1, AE_EPOCHS + 1):
    ae.train(); opt_ae.zero_grad()
    out = ae(Xt); loss = nn.MSELoss()(out, Xt)
    loss.backward(); opt_ae.step()
    ae_losses.append(loss.item())

# Train VAE
vae = VAE(input_dim=WINDOW, bottleneck=BOTTLENECK)
opt_vae = torch.optim.Adam(vae.parameters(), lr=LR)
vae_losses = []
print(f"  Training VAE ({AE_EPOCHS} epochs) ...")
for ep in range(1, AE_EPOCHS + 1):
    vae.train(); opt_vae.zero_grad()
    recon_batch, mu, logvar = vae(Xt)
    loss = vae_loss_function(recon_batch, Xt, mu, logvar)
    loss.backward(); opt_vae.step()
    vae_losses.append(loss.item() / len(Xt))

# Eval
ae.eval(); vae.eval()
with torch.no_grad():
    X_ae_n = ae(Xt).numpy()
    X_vae_n, _, _ = vae(Xt)
    X_vae_n = X_vae_n.numpy()

T_ae = windows_to_signal(scaler.inverse_transform(X_ae_n), n)
T_vae = windows_to_signal(scaler.inverse_transform(X_vae_n), n)
rmse_ae = float(np.sqrt(mean_squared_error(T_cal, T_ae)))
rmse_vae = float(np.sqrt(mean_squared_error(T_cal, T_vae)))
ratio_ae = WINDOW / BOTTLENECK
ratio_vae = WINDOW / BOTTLENECK

print(f"\n  Autoencoder RMSE : {rmse_ae:.2f} C")
print(f"  VAE RMSE         : {rmse_vae:.2f} C")

torch.save(ae.state_dict(), "autoencoder_compressor.pth")
torch.save(vae.state_dict(), "vae_compressor.pth")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("COMPARISON TABLE -- All compression methods (ATP-3)")
print("=" * 65)

df_compare = pd.DataFrame({
    "Method"    : ["Wavelet", "SVD", "Delta", "PCA", "Autoencoder", "VAE"],
    "RMSE_C"    : [round(rmse_wav,2), round(rmse_svd,2), round(rmse_delta,4), 
                   round(rmse_pca,2), round(rmse_ae,2), round(rmse_vae,2)],
    "Ratio"     : [round(ratio_wav,1), round(ratio_svd,1), round(ratio_delta,1), 
                   round(ratio_pca,1), round(ratio_ae,1), round(ratio_vae,1)],
    "Type"      : ["Baseline", "Baseline", "Near-Lossless", "ML", "ML", "ML-Prob"]
})
print(df_compare.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# 5-ROW PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("5-ROW PREVIEW -- hottest region")
print("=" * 65)

df_prev = pd.DataFrame({
    "Time_s"      : np.round(time_s[idx], 4),
    "Calibrated"  : np.round(T_cal[idx], 2),
    "Delta"       : np.round(T_delta[idx], 2),
    "PCA"         : np.round(T_pca[idx], 2),
    "Autoencoder" : np.round(T_ae[idx], 2),
    "VAE"         : np.round(T_vae[idx], 2),
}, index=[f"t{i}" for i in idx])
print(df_prev.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Generating visualisation plots ...")
print("=" * 65)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("D3 -- ML Compression: VAE, AE, PCA vs Baseline\n"
             "NIST Layer01 IN625 data", fontsize=13, fontweight="bold")

# Panel A -- full signal
ax = axes[0, 0]
ax.plot(time_s, T_cal,  color="black",      lw=1.2, label="Calibrated (original)")
ax.plot(time_s, T_delta,color="red",        lw=1.0, label=f"Delta RMSE={rmse_delta:.2f}C")
ax.plot(time_s, T_pca,  color="darkorange", lw=1.0, label=f"PCA RMSE={rmse_pca:.1f}C")
ax.plot(time_s, T_vae,  color="green",      lw=1.0, ls="--", label=f"VAE RMSE={rmse_vae:.1f}C")
ax.set_title("A  Full signal -- original vs reconstructed")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)"); ax.legend(fontsize=8)

# Panel B -- zoomed at peak
Z1, Z2 = max(0, HOT-80), min(n, HOT+80)
ax = axes[0, 1]
ax.plot(time_s[Z1:Z2], T_cal[Z1:Z2], color="black",      lw=1.5, label="Calibrated")
ax.plot(time_s[Z1:Z2], T_delta[Z1:Z2], color="red",        lw=1.0, label="Delta")
ax.plot(time_s[Z1:Z2], T_pca[Z1:Z2], color="darkorange", lw=1.2, label="PCA")
ax.plot(time_s[Z1:Z2], T_vae[Z1:Z2],  color="green",      lw=1.2, label="VAE", ls="--")
ax.set_title("B  Zoomed at peak temperature")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)"); ax.legend(fontsize=8)

# Panel C -- training losses
ax = axes[1, 0]
ax.plot(ae_losses, color="blue", lw=1.5, label="AE loss")
ax.plot(vae_losses, color="green", lw=1.5, label="VAE loss")
ax.set_title("C  Training losses (lower = better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=8); ax.set_yscale("log")

# Panel D -- RMSE vs ratio bar chart
ax    = axes[1, 1]
methods = ["Wavelet", "SVD", "Delta", "PCA", "AE", "VAE"]
rmses   = [rmse_wav, rmse_svd, rmse_delta, rmse_pca, rmse_ae, rmse_vae]
ratios  = [ratio_wav, ratio_svd, ratio_delta, ratio_pca, ratio_ae, ratio_ae]
colors  = ["steelblue", "cornflowerblue", "red", "darkorange", "blue", "green"]
x = np.arange(len(methods)); w = 0.35
ax.bar(x - w/2, rmses, w, color=colors, alpha=0.8, label="RMSE (C)")
ax2 = ax.twinx()
ax2.bar(x + w/2, ratios, w, color=colors, alpha=0.3, hatch="//", label="Ratio")
ax.set_title("D  RMSE vs Compression ratio")
ax.set_ylabel("RMSE (C)"); ax2.set_ylabel("Ratio")
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.savefig("ml_compress_result.png", dpi=150)
print("  Plot saved -> ml_compress_result.png")

# =============================================================================
# EXTRA MODELS — Added based on supervisor feedback (Amit & Karthikeyan)
# Model 3 : Variational Autoencoder (VAE)
# Model 4 : Deep Autoencoder (deeper architecture)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — VARIATIONAL AUTOENCODER (VAE)
# Architecture:
#   Encoder: Linear(32->16->8) -> mu + log_var (bottleneck=3)
#   Decoder: Linear(3->8->16->32)
# VAE learns a probabilistic compressed representation
# Better generalisation than standard Autoencoder
# Used in advanced signal compression and anomaly detection
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MODEL 3 -- Variational Autoencoder (VAE)")
print(f"  Architecture: {WINDOW}->16->8->mu/logvar({BOTTLENECK})->{WINDOW}")
print("  Learns probabilistic compressed representation")
print("=" * 65)

class VAE(nn.Module):
    """
    Variational Autoencoder for signal compression.
    Encoder outputs mean (mu) and log-variance (log_var)
    instead of a single compressed value.
    The bottleneck samples from N(mu, exp(log_var)).
    This forces the compressed space to be smooth and continuous,
    leading to better generalisation than standard autoencoder.
    KL divergence term in loss keeps compressed space well-structured.
    """
    def __init__(self, input_dim=32, bottleneck=3):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(8, bottleneck)
        self.fc_logvar = nn.Linear(8, bottleneck)
        self.decoder   = nn.Sequential(
            nn.Linear(bottleneck, 8),  nn.ReLU(),
            nn.Linear(8, 16),          nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def encode(self, x):
        h      = self.encoder_shared(x)
        mu     = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterise(self, mu, log_var):
        """Sample from N(mu, exp(log_var)) using reparameterisation trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z    = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

def vae_loss(recon, x, mu, log_var):
    """VAE loss = reconstruction loss + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + 0.001 * kl_loss   # small KL weight for stability

vae      = VAE(input_dim=WINDOW, bottleneck=BOTTLENECK)
opt_vae  = torch.optim.Adam(vae.parameters(), lr=LR)
vae_losses = []

print(f"\n  Training VAE for {AE_EPOCHS} epochs ...")
for ep in range(1, AE_EPOCHS + 1):
    vae.train(); opt_vae.zero_grad()
    recon, mu, log_var = vae(Xt)
    loss = vae_loss(recon, Xt, mu, log_var)
    loss.backward(); opt_vae.step()
    vae_losses.append(loss.item())
    if ep % 20 == 0 or ep == 1:
        print(f"    Epoch {ep:3d}/{AE_EPOCHS}  Loss: {loss.item():.6f}")

vae.eval()
with torch.no_grad():
    X_vae_n, _, _ = vae(Xt)
    X_vae_n = X_vae_n.numpy()
X_vae_orig = scaler.inverse_transform(X_vae_n)
T_vae      = windows_to_signal(X_vae_orig, n)
rmse_vae   = float(np.sqrt(mean_squared_error(T_cal, T_vae)))
ratio_vae  = WINDOW / BOTTLENECK

print(f"\n  RMSE              : {rmse_vae:.2f} C")
print(f"  Compression ratio : {ratio_vae:.1f}x")
print(f"  Improvement vs SVD: {(1-rmse_vae/rmse_svd)*100:.1f}%")

torch.save(vae.state_dict(), "vae_compressor.pth")
print("  Model saved --> vae_compressor.pth")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — DEEP AUTOENCODER (deeper architecture)
# Architecture:
#   Encoder: Linear(32->24->16->8->3)
#   Decoder: Linear(3->8->16->24->32)
# More layers = can learn more complex signal patterns
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("MODEL 4 -- Deep Autoencoder (deeper architecture)")
print(f"  Encoder: {WINDOW}->24->16->8->{BOTTLENECK}")
print(f"  Decoder: {BOTTLENECK}->8->16->24->{WINDOW}")
print("=" * 65)

class DeepAutoencoder(nn.Module):
    """
    Deeper autoencoder with more hidden layers.
    Can learn more complex non-linear signal structures
    compared to the standard 3-layer autoencoder.
    """
    def __init__(self, input_dim=32, bottleneck=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24), nn.ReLU(),
            nn.Linear(24, 16),        nn.ReLU(),
            nn.Linear(16, 8),         nn.ReLU(),
            nn.Linear(8, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 8), nn.ReLU(),
            nn.Linear(8, 16),         nn.ReLU(),
            nn.Linear(16, 24),        nn.ReLU(),
            nn.Linear(24, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

dae       = DeepAutoencoder(input_dim=WINDOW, bottleneck=BOTTLENECK)
opt_dae   = torch.optim.Adam(dae.parameters(), lr=LR)
crit_dae  = nn.MSELoss()
dae_losses = []

print(f"\n  Training Deep Autoencoder for {AE_EPOCHS} epochs ...")
for ep in range(1, AE_EPOCHS + 1):
    dae.train(); opt_dae.zero_grad()
    out  = dae(Xt); loss = crit_dae(out, Xt)
    loss.backward(); opt_dae.step()
    dae_losses.append(loss.item())
    if ep % 20 == 0 or ep == 1:
        print(f"    Epoch {ep:3d}/{AE_EPOCHS}  Loss: {loss.item():.6f}")

dae.eval()
with torch.no_grad():
    X_dae_n = dae(Xt).numpy()
X_dae_orig = scaler.inverse_transform(X_dae_n)
T_dae      = windows_to_signal(X_dae_orig, n)
rmse_dae   = float(np.sqrt(mean_squared_error(T_cal, T_dae)))
ratio_dae  = WINDOW / BOTTLENECK

print(f"\n  RMSE              : {rmse_dae:.2f} C")
print(f"  Compression ratio : {ratio_dae:.1f}x")
print(f"  Improvement vs SVD: {(1-rmse_dae/rmse_svd)*100:.1f}%")

torch.save(dae.state_dict(), "deep_autoencoder_compressor.pth")
print("  Model saved --> deep_autoencoder_compressor.pth")


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED COMPARISON TABLE — All 6 methods
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("UPDATED COMPARISON TABLE -- All 6 compression methods")
print("=" * 65)

df_full = pd.DataFrame({
    "Method"    : [
        "Wavelet (baseline)",
        "SVD (baseline)",
        "PCA",
        "Autoencoder",
        "VAE",
        "Deep Autoencoder",
    ],
    "Type"      : ["Classical","Classical","ML","ML","ML","ML"],
    "RMSE_C"    : [
        round(rmse_wav, 2),
        round(rmse_svd, 2),
        round(rmse_pca, 2),
        round(rmse_ae,  2),
        round(rmse_vae, 2),
        round(rmse_dae, 2),
    ],
    "Ratio"     : [
        round(ratio_wav, 1),
        round(ratio_svd, 1),
        round(ratio_pca, 1),
        round(ratio_ae,  1),
        round(ratio_vae, 1),
        round(ratio_dae, 1),
    ],
})
print(df_full.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED 5-ROW PREVIEW — All 6 methods
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("5-ROW PREVIEW -- all 6 methods at hottest region")
print("=" * 65)

df_prev2 = pd.DataFrame({
    "Time_s"     : np.round(time_s[idx], 4),
    "Calibrated" : np.round(T_cal[idx],  2),
    "Wavelet"    : np.round(T_wav[idx],  2),
    "PCA"        : np.round(T_pca[idx],  2),
    "Autoenc"    : np.round(T_ae[idx],   2),
    "VAE"        : np.round(T_vae[idx],  2),
    "DeepAE"     : np.round(T_dae[idx],  2),
}, index=[f"t{i}" for i in idx])
print(df_prev2.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# UPDATED VISUALISATION — All 6 methods
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("Generating updated plots (all 6 methods) ...")
print("=" * 65)

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle("D3 -- ML Compression: AE + VAE + Deep AE + PCA\n"
              "NIST Layer01 IN625 data", fontsize=13, fontweight="bold")

# Panel A: Full signal
ax = axes2[0, 0]
ax.plot(time_s, T_cal,  color="black",      lw=1.2, label="Calibrated (original)")
ax.plot(time_s, T_wav,  color="lightblue",  lw=0.9, alpha=0.9, label=f"Wavelet {ratio_wav:.1f}x RMSE={rmse_wav:.1f}C")
ax.plot(time_s, T_pca,  color="darkorange", lw=0.9, label=f"PCA {ratio_pca:.1f}x RMSE={rmse_pca:.1f}C")
ax.plot(time_s, T_ae,   color="green",      lw=0.9, ls="--", label=f"AE {ratio_ae:.1f}x RMSE={rmse_ae:.1f}C")
ax.plot(time_s, T_vae,  color="purple",     lw=0.9, ls="-.", label=f"VAE {ratio_vae:.1f}x RMSE={rmse_vae:.1f}C")
ax.plot(time_s, T_dae,  color="brown",      lw=0.9, ls=":", label=f"DeepAE {ratio_dae:.1f}x RMSE={rmse_dae:.1f}C")
ax.set_title("A  Full signal -- all 6 methods")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel B: Zoomed
Z1, Z2 = max(0, HOT-80), min(n, HOT+80)
ax = axes2[0, 1]
ax.plot(time_s[Z1:Z2], T_cal[Z1:Z2],  color="black",      lw=1.5, label="Calibrated")
ax.plot(time_s[Z1:Z2], T_wav[Z1:Z2],  color="lightblue",  lw=1.0, label="Wavelet")
ax.plot(time_s[Z1:Z2], T_pca[Z1:Z2],  color="darkorange", lw=1.1, label="PCA")
ax.plot(time_s[Z1:Z2], T_ae[Z1:Z2],   color="green",      lw=1.1, ls="--", label="AE")
ax.plot(time_s[Z1:Z2], T_vae[Z1:Z2],  color="purple",     lw=1.1, ls="-.", label="VAE")
ax.plot(time_s[Z1:Z2], T_dae[Z1:Z2],  color="brown",      lw=1.1, ls=":", label="DeepAE")
ax.set_title("B  Zoomed at peak temperature")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# Panel C: Training losses
ax = axes2[1, 0]
ax.plot(ae_losses,  color="green",  lw=1.4, label="Autoencoder")
ax.plot(vae_losses, color="purple", lw=1.4, ls="--", label="VAE")
ax.plot(dae_losses, color="brown",  lw=1.4, ls="-.", label="Deep AE")
ax.set_title("C  Training loss curves (lower=better)")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.legend(fontsize=8); ax.set_yscale("log"); ax.grid(True, alpha=0.3)

# Panel D: RMSE vs Ratio scatter
ax = axes2[1, 1]
methods_sc = ["Wavelet","SVD","PCA","AE","VAE","DeepAE"]
rmse_sc    = [rmse_wav, rmse_svd, rmse_pca, rmse_ae, rmse_vae, rmse_dae]
ratio_sc   = [ratio_wav, ratio_svd, ratio_pca, ratio_ae, ratio_vae, ratio_dae]
colors_sc  = ["lightblue","cornflowerblue","darkorange","green","purple","brown"]
ax.scatter(ratio_sc, rmse_sc, c=colors_sc, s=150, zorder=5,
           edgecolors="black", lw=0.8)
for m, x, y in zip(methods_sc, ratio_sc, rmse_sc):
    ax.annotate(m, (x, y), textcoords="offset points",
                xytext=(6, 5), fontsize=9)
ax.axvline(x=4.0, color="green", ls="--", lw=1.2, label="4x target")
ax.set_title("D  RMSE vs Ratio (bottom-right = best trade-off)")
ax.set_xlabel("Compression ratio (higher=smaller file)")
ax.set_ylabel("RMSE (C) -- lower=better")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ml_compress_result_updated.png", dpi=150, bbox_inches="tight")
print("  Plot saved --> ml_compress_result_updated.png")
plt.show()

print()
print("=" * 65)
print("ALL 4 ML COMPRESSION MODELS COMPLETE")
print("=" * 65)
print(f"  Wavelet (baseline) : RMSE={rmse_wav:.2f}C  ratio={ratio_wav:.1f}x")
print(f"  SVD (baseline)     : RMSE={rmse_svd:.2f}C  ratio={ratio_svd:.1f}x")
print(f"  PCA                : RMSE={rmse_pca:.2f}C  ratio={ratio_pca:.1f}x")
print(f"  Autoencoder        : RMSE={rmse_ae:.2f}C  ratio={ratio_ae:.1f}x")
print(f"  VAE                : RMSE={rmse_vae:.2f}C  ratio={ratio_vae:.1f}x")
print(f"  Deep Autoencoder   : RMSE={rmse_dae:.2f}C  ratio={ratio_dae:.1f}x")
print()
print("  Compare all ML results against classical methods.")
print("=" * 65)