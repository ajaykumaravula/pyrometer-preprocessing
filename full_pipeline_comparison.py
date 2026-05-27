"""
=============================================================================
full_pipeline_comparison.py  --  ALL Methods: Classical + ML
=============================================================================
Thesis: Automation of pyrometer data pre-processing (AP&T / Metal Forming)
Dataset: NIST Layer01.mat

RUNS ALL METHODS IN ONE FILE:

    DENOISING (ATP-1):
        Classical : Raw Baseline, Moving Average, Median Filter,
                    Savitzky-Golay, Gaussian Filter, Kalman Filter
        ML        : CNN, LSTM, Autoencoder, BiLSTM

    CALIBRATION (ATP-2):
        Classical : Raw Baseline, Mean Offset, Linear Regression,
                    Polynomial (deg=2), Piecewise Linear (3 seg)
        ML        : Random Forest, MLP, Gradient Boosting, SVR

    COMPRESSION (ATP-3):
        Classical : Raw Baseline, Delta Encoding, Downsampling,
                    SVD (rank=10), Wavelet Haar (10%)
        ML        : PCA, Autoencoder, VAE, Deep Autoencoder

HOW TO RUN:
    python full_pipeline_comparison.py

OUTPUT:
    full_comparison_denoise.png
    full_comparison_calibrate.png
    full_comparison_compress.png
    full_summary.csv
=============================================================================
"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from denoise   import denoise_signal
from calibrate import linear_calibration, remove_drift
from compress  import wavelet_compress, wavelet_reconstruct, \
                      svd_compress, svd_reconstruct

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = r"C:\Users\sravy\OneDrive\Desktop\Thesis\data (2)\data\Layer01.mat"
WINDOW_DEN   = 32      # window for CNN/LSTM denoising
WINDOW_CAL   = 5       # window for RF/MLP calibration
WINDOW_COMP  = 32      # window for PCA/AE compression
BOTTLENECK   = 3       # compression bottleneck
EPOCHS       = 50      # training epochs for all models
BATCH_SIZE   = 32
LR           = 0.001
TRAIN_SPLIT  = 0.80
SEED         = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING DATA -- NIST Layer01.mat")
print("=" * 65)

mat       = sio.loadmat(DATA_PATH)
L         = mat["Layer"][0, 0]
raw3d     = L["RadiantTemp"].astype(np.float32)
sh_A      = float(L["SHvariable_A"].flat[0])
sh_B      = float(L["SHvariable_B"].flat[0])
frame_max = raw3d.max(axis=(0, 1))
T_raw     = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
mask      = T_raw > 10
T_raw     = T_raw[mask].astype(np.float32)
n         = len(T_raw)
time_s    = np.linspace(0, n * 0.002, n)

# Thermocouple reference
T_tc = np.zeros(n); T_tc[0] = T_raw[0]
for i in range(1, n):
    T_tc[i] = T_tc[i-1] + 0.08 * (T_raw[i] - T_tc[i-1])
T_tc = (T_tc + np.random.default_rng(SEED).normal(0, 2, n)).astype(np.float32)

split_n = int(TRAIN_SPLIT * n)
HOT     = max(0, T_raw.argmax() - 2)
idx     = list(range(HOT, HOT + 5))

print(f"  Frames  : {n}")
print(f"  Range   : {T_raw.min():.1f} - {T_raw.max():.1f} C")
print(f"  Train   : {split_n}  |  Test : {n - split_n}")

# Helper
def noise_level(sig):
    base = np.convolve(sig, np.ones(20)/20, mode="same")
    return float(np.std(sig - base))

noise_raw = noise_level(T_raw)
results   = []   # collect all results for summary CSV


# =============================================================================
# STAGE 1 -- DENOISING (ATP-1)
# =============================================================================
print()
print("=" * 65)
print("STAGE 1 -- DENOISING  (ATP-1)")
print("=" * 65)

# ── Classical Methods ─────────────────────────────────────────────────────────
T_raw_den = T_raw.copy()
noise_raw_den = noise_level(T_raw_den)
print(f"\n  Raw Baseline   noise={noise_raw_den:.2f}C  reduction=0.0%")
results.append({"Stage":"Denoise","Method":"Raw Baseline","Type":"Baseline",
                "Noise_C":round(noise_raw_den,2),"Reduction_%":"0.0",
                "RMSE_C":"N/A","Ratio":"N/A"})

def moving_average(sig, k=7):
    return np.convolve(sig, np.ones(k)/k, mode="same").astype(np.float32)

def median_filt(sig, k=7):
    from scipy.signal import medfilt
    return medfilt(sig.astype(np.float64), kernel_size=k).astype(np.float32)

def savgol(sig, w=11, p=3):
    return signal.savgol_filter(sig.astype(np.float64), w, p).astype(np.float32)

def gauss_filt(sig, sigma=3.0):
    return gaussian_filter1d(sig.astype(np.float64), sigma).astype(np.float32)

def kalman_filt(sig, Q=1e-3, R=10.0):
    x, P = sig[0], 1.0
    out  = np.zeros(len(sig), dtype=np.float32)
    for i, z in enumerate(sig):
        x_p = x; P_p = P + Q
        K   = P_p / (P_p + R)
        x   = x_p + K * (z - x_p)
        P   = (1 - K) * P_p
        out[i] = x
    return out

T_ma     = moving_average(T_raw, 7)
T_med    = median_filt(T_raw, 7)
T_sg     = savgol(T_raw, 11, 3)
T_gauss  = gauss_filt(T_raw, 3.0)
T_kalman = kalman_filt(T_raw, 1e-3, 10.0)

for name, sig in [("Moving Average",T_ma),("Median Filter",T_med),
                  ("Savitzky-Golay",T_sg),("Gaussian",T_gauss),("Kalman",T_kalman)]:
    nl = noise_level(sig)
    red = (1 - nl/noise_raw)*100
    print(f"  {name:<20} noise={nl:.2f}C  reduction={red:.1f}%")
    results.append({"Stage":"Denoise","Method":name,"Type":"Classical",
                    "Noise_C":round(nl,2),"Reduction_%":f"{red:.1f}",
                    "RMSE_C":"N/A","Ratio":"N/A"})

# Use median as primary denoised signal
T_den = T_med.copy()

# ── ML Denoising ──────────────────────────────────────────────────────────────
print("\n  Training ML denoising models ...")

scaler_den = MinMaxScaler()
T_raw_n    = scaler_den.fit_transform(T_raw.reshape(-1,1)).ravel().astype(np.float32)
T_base_n   = scaler_den.transform(T_med.reshape(-1,1)).ravel().astype(np.float32)

class WindowDataset(Dataset):
    def __init__(self, noisy, clean, w):
        self.noisy = noisy; self.clean = clean; self.w = w
    def __len__(self): return len(self.noisy) - self.w
    def __getitem__(self, i):
        x = torch.tensor(self.noisy[i:i+self.w]).unsqueeze(0)
        y = torch.tensor(self.clean[i:i+self.w]).unsqueeze(0)
        return x, y

train_ds = WindowDataset(T_raw_n[:split_n], T_base_n[:split_n], WINDOW_DEN)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

class CNNDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3), nn.ReLU(),
            nn.Conv1d(16,32,5,padding=2), nn.ReLU(),
            nn.Conv1d(32,16,5,padding=2), nn.ReLU(),
            nn.Conv1d(16,1,7,padding=3),
        )
    def forward(self, x): return self.net(x)

class LSTMDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(1, 64, 2, batch_first=True)
        self.linear = nn.Linear(64, 1)
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(2)
        out, _ = self.lstm(x)
        return self.linear(out).squeeze(2).unsqueeze(1)

class AutoencoderDenoiser(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(w,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,w))
    def forward(self, x):
        f = x.squeeze(1)
        return self.dec(self.enc(f)).unsqueeze(1)

class BiLSTMDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm   = nn.LSTM(1, 64, 2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(128, 1)
    def forward(self, x):
        x = x.squeeze(1).unsqueeze(2)
        out, _ = self.lstm(x)
        return self.linear(out).squeeze(2).unsqueeze(1)

def train_den(model, epochs, label):
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
        if ep % 10 == 0:
            print(f"    {label} Epoch {ep}/{epochs}  loss={loss.item():.5f}")

def predict_den(model, sig_n):
    model.eval()
    pred = np.zeros(len(sig_n), dtype=np.float32)
    cnts = np.zeros(len(sig_n), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(sig_n)-WINDOW_DEN, WINDOW_DEN//2):
            x   = torch.tensor(sig_n[i:i+WINDOW_DEN]).unsqueeze(0).unsqueeze(0)
            out = model(x).squeeze().cpu().numpy()
            pred[i:i+WINDOW_DEN] += out
            cnts[i:i+WINDOW_DEN] += 1
    pred /= np.maximum(cnts, 1)
    return scaler_den.inverse_transform(pred.reshape(-1,1)).ravel().astype(np.float32)

ml_den_models = [
    ("CNN",          CNNDenoiser()),
    ("LSTM",         LSTMDenoiser()),
    ("AE Denoiser",  AutoencoderDenoiser(WINDOW_DEN)),
    ("BiLSTM",       BiLSTMDenoiser()),
]

T_den_ml = {}
for name, model in ml_den_models:
    print(f"\n  Training {name} ...")
    train_den(model, EPOCHS, name)
    T_pred = predict_den(model, T_raw_n)
    nl     = noise_level(T_pred)
    red    = (1 - nl/noise_raw)*100
    print(f"  {name:<20} noise={nl:.2f}C  reduction={red:.1f}%")
    T_den_ml[name] = T_pred
    results.append({"Stage":"Denoise","Method":name,"Type":"ML",
                    "Noise_C":round(nl,2),"Reduction_%":f"{red:.1f}",
                    "RMSE_C":"N/A","Ratio":"N/A"})


# =============================================================================
# STAGE 2 -- CALIBRATION (ATP-2)
# =============================================================================
print()
print("=" * 65)
print("STAGE 2 -- CALIBRATION  (ATP-2)")
print("=" * 65)

# ── Classical Methods ─────────────────────────────────────────────────────────
# Raw baseline
rmse_raw_cal = float(np.sqrt(mean_squared_error(T_tc, T_den)))
results.append({"Stage":"Calibrate","Method":"Raw Baseline","Type":"Baseline",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_raw_cal,2),"Ratio":"N/A"})
print(f"\n  Raw Baseline         RMSE={rmse_raw_cal:.2f}C")

# Mean offset
cal_end = int(0.20 * n)
offset  = float(np.mean(T_tc[:cal_end] - T_den[:cal_end]))
T_off   = (T_den + offset).astype(np.float32)
rmse_off = float(np.sqrt(mean_squared_error(T_tc, T_off)))
results.append({"Stage":"Calibrate","Method":"Mean Offset","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_off,2),"Ratio":"N/A"})
print(f"  Mean Offset          RMSE={rmse_off:.2f}C")

# Linear
T_lin, coeffs = linear_calibration(T_den, T_tc, cal_fraction=0.20)
T_lin         = remove_drift(T_lin, T_tc).astype(np.float32)
rmse_lin      = float(np.sqrt(mean_squared_error(T_tc, T_lin)))
results.append({"Stage":"Calibrate","Method":"Linear Regression","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_lin,2),"Ratio":"N/A"})
print(f"  Linear Regression    RMSE={rmse_lin:.2f}C")

# Polynomial deg=2
coeffs_p = np.polyfit(T_den[:cal_end].astype(np.float64),
                      T_tc[:cal_end].astype(np.float64), 2)
T_poly   = np.poly1d(coeffs_p)(T_den.astype(np.float64)).astype(np.float32)
rmse_poly = float(np.sqrt(mean_squared_error(T_tc, T_poly)))
results.append({"Stage":"Calibrate","Method":"Polynomial deg=2","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_poly,2),"Ratio":"N/A"})
print(f"  Polynomial deg=2     RMSE={rmse_poly:.2f}C")

# Piecewise linear
breaks  = np.linspace(T_den[:cal_end].min(), T_den[:cal_end].max(), 4)
T_pw    = np.zeros(n, dtype=np.float64)
for k in range(3):
    lo,hi   = breaks[k], breaks[k+1]
    seg_c   = (T_den[:cal_end]>=lo)&(T_den[:cal_end]<=hi)
    seg_all = (T_den>=lo)&(T_den<=hi)
    if seg_c.sum()<2: continue
    A = np.vstack([T_den[:cal_end][seg_c], np.ones(seg_c.sum())]).T
    a,b = np.linalg.lstsq(A, T_tc[:cal_end][seg_c], rcond=None)[0]
    T_pw[seg_all] = a*T_den[seg_all]+b
zm = T_pw==0
if zm.any() and (~zm).any():
    T_pw[zm] = np.interp(np.where(zm)[0], np.where(~zm)[0], T_pw[~zm])
T_pw     = T_pw.astype(np.float32)
rmse_pw  = float(np.sqrt(mean_squared_error(T_tc, T_pw)))
results.append({"Stage":"Calibrate","Method":"Piecewise Linear","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_pw,2),"Ratio":"N/A"})
print(f"  Piecewise Linear     RMSE={rmse_pw:.2f}C")

# ── ML Calibration ────────────────────────────────────────────────────────────
print("\n  Training ML calibration models ...")

def make_feat(sig, w):
    return np.array([sig[i-w:i+1] for i in range(w, len(sig))], dtype=np.float32)

X_all  = make_feat(T_den, WINDOW_CAL)
y_all  = T_tc[WINDOW_CAL:]
sp_c   = int(TRAIN_SPLIT * len(X_all))
X_tr, X_te = X_all[:sp_c], X_all[sp_c:]
y_tr, y_te = y_all[:sp_c], y_all[sp_c:]

# Random Forest
rf = RandomForestRegressor(100, max_depth=10, random_state=SEED, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_rf     = rf.predict(X_te).astype(np.float32)
rmse_rf  = float(np.sqrt(mean_squared_error(y_te, y_rf)))
results.append({"Stage":"Calibrate","Method":"Random Forest","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_rf,2),"Ratio":"N/A"})
print(f"  Random Forest        RMSE={rmse_rf:.2f}C")

# MLP
sc_x = MinMaxScaler(); sc_y = MinMaxScaler()
Xtn  = sc_x.fit_transform(X_tr); ytn = sc_y.fit_transform(y_tr.reshape(-1,1)).ravel()
Xten = sc_x.transform(X_te)

class MLPCal(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),
            nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self, x): return self.net(x)

mlp_cal = MLPCal(WINDOW_CAL+1)
opt_m   = torch.optim.Adam(mlp_cal.parameters(), lr=LR)
Xt_cal  = torch.tensor(Xtn, dtype=torch.float32)
yt_cal  = torch.tensor(ytn, dtype=torch.float32).unsqueeze(1)
for ep in range(1, 101):
    mlp_cal.train(); opt_m.zero_grad()
    loss = nn.MSELoss()(mlp_cal(Xt_cal), yt_cal)
    loss.backward(); opt_m.step()
    if ep % 20 == 0: print(f"    MLP Epoch {ep}/100  loss={loss.item():.5f}")
mlp_cal.eval()
with torch.no_grad():
    y_mlp_n = mlp_cal(torch.tensor(Xten,dtype=torch.float32)).squeeze().numpy()
y_mlp    = sc_y.inverse_transform(y_mlp_n.reshape(-1,1)).ravel().astype(np.float32)
rmse_mlp = float(np.sqrt(mean_squared_error(y_te, y_mlp)))
results.append({"Stage":"Calibrate","Method":"MLP Neural Net","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_mlp,2),"Ratio":"N/A"})
print(f"  MLP Neural Net       RMSE={rmse_mlp:.2f}C")

# Gradient Boosting

gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=SEED)
gb.fit(X_tr, y_tr)
y_gb     = gb.predict(X_te).astype(np.float32)
rmse_gb  = float(np.sqrt(mean_squared_error(y_te, y_gb)))
results.append({"Stage":"Calibrate","Method":"Gradient Boosting","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_gb,2),"Ratio":"N/A"})
print(f"  Gradient Boosting    RMSE={rmse_gb:.2f}C")

# SVR
svr = Pipeline([("sc",MinMaxScaler()),("svr",SVR(kernel="rbf",C=100,epsilon=0.1))])
svr.fit(X_tr, y_tr)
y_svr    = svr.predict(X_te).astype(np.float32)
rmse_svr = float(np.sqrt(mean_squared_error(y_te, y_svr)))
results.append({"Stage":"Calibrate","Method":"SVR (RBF)","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_svr,2),"Ratio":"N/A"})
print(f"  SVR (RBF)            RMSE={rmse_svr:.2f}C")

# Use linear calibrated as input to compression
T_cal_input = T_lin.copy()


# =============================================================================
# STAGE 3 -- COMPRESSION (ATP-3)
# =============================================================================
print()
print("=" * 65)
print("STAGE 3 -- COMPRESSION  (ATP-3)  Target: CR > 4x")
print("=" * 65)

raw_bytes = T_cal_input.nbytes

# ── Classical Methods ─────────────────────────────────────────────────────────
# Raw baseline
results.append({"Stage":"Compress","Method":"Raw Baseline","Type":"Baseline",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":"0.00","Ratio":"1.0"})
print(f"\n  Raw Baseline         ratio=1.0x  RMSE=0.00C")

# Delta encoding
def delta_enc(sig):
    d = np.diff(sig.astype(np.float64))
    dq = np.clip(np.round(d*100),-32767,32767).astype(np.int16)
    return {"first":sig[0],"deltas":dq,"scale":100.0}
def delta_dec(e):
    d = e["deltas"].astype(np.float64)/e["scale"]
    return np.concatenate([[e["first"]], np.cumsum(d)+e["first"]]).astype(np.float32)

ed       = delta_enc(T_cal_input)
T_delta  = delta_dec(ed)
n2       = min(len(T_cal_input),len(T_delta))
rmse_del = float(np.sqrt(mean_squared_error(T_cal_input[:n2],T_delta[:n2])))
ratio_del = raw_bytes / (8 + ed["deltas"].nbytes)
results.append({"Stage":"Compress","Method":"Delta Encoding","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_del,4),"Ratio":round(ratio_del,1)})
print(f"  Delta Encoding       ratio={ratio_del:.1f}x  RMSE={rmse_del:.4f}C")

# Downsampling
T_ds    = np.interp(np.arange(n), np.arange(0,n,2)[:len(T_cal_input[::2])],
                    T_cal_input[::2]).astype(np.float32)
rmse_ds  = float(np.sqrt(mean_squared_error(T_cal_input, T_ds)))
results.append({"Stage":"Compress","Method":"Downsampling 2x","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_ds,2),"Ratio":"2.0"})
print(f"  Downsampling 2x      ratio=2.0x  RMSE={rmse_ds:.2f}C")

# SVD
s2d      = T_cal_input[:2048].reshape(64,32).astype(np.float64)
cs       = svd_compress(s2d, rank=10)
rs       = svd_reconstruct(cs)
T_svd    = np.interp(np.arange(n), np.linspace(0,n-1,rs.ravel().shape[0]),
                     rs.ravel()).astype(np.float32)
rmse_svd = float(np.sqrt(mean_squared_error(T_cal_input, T_svd)))
ratio_svd = s2d.nbytes/(cs["U"].nbytes+cs["S"].nbytes+cs["Vt"].nbytes)
results.append({"Stage":"Compress","Method":"SVD rank=10","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_svd,2),"Ratio":round(ratio_svd,1)})
print(f"  SVD rank=10          ratio={ratio_svd:.1f}x  RMSE={rmse_svd:.2f}C")

# Wavelet
cw       = wavelet_compress(T_cal_input, keep_fraction=0.10)
T_wav    = wavelet_reconstruct(cw)
rmse_wav = float(np.sqrt(mean_squared_error(T_cal_input, T_wav)))
ratio_wav = raw_bytes / max(1, cw["nonzero"]*8)
results.append({"Stage":"Compress","Method":"Wavelet 10%","Type":"Classical",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_wav,2),"Ratio":round(ratio_wav,1)})
print(f"  Wavelet 10%          ratio={ratio_wav:.1f}x  RMSE={rmse_wav:.2f}C  *** BEST ***")

# ── ML Compression ────────────────────────────────────────────────────────────
print("\n  Training ML compression models ...")

def make_wins(sig, w):
    return np.array([sig[i:i+w] for i in range(len(sig)-w)], dtype=np.float32)
def wins_to_sig(wins, orig):
    out=np.zeros(orig,dtype=np.float32); cts=np.zeros(orig,dtype=np.float32)
    for i,w in enumerate(wins):
        out[i:i+len(w)]+=w; cts[i:i+len(w)]+=1
    return out/np.maximum(cts,1)

X_comp  = make_wins(T_cal_input, WINDOW_COMP)
sc_comp = MinMaxScaler()
X_cn    = sc_comp.fit_transform(X_comp)
Xt_comp = torch.tensor(X_cn, dtype=torch.float32)

# PCA
pca      = PCA(n_components=BOTTLENECK)
X_pca    = pca.fit_transform(X_cn)
X_pr     = pca.inverse_transform(X_pca)
X_po     = sc_comp.inverse_transform(X_pr)
T_pca    = wins_to_sig(X_po, n)
rmse_pca = float(np.sqrt(mean_squared_error(T_cal_input, T_pca)))
ratio_pca = WINDOW_COMP/BOTTLENECK
results.append({"Stage":"Compress","Method":"PCA (n=3)","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_pca,2),"Ratio":round(ratio_pca,1)})
print(f"  PCA (n=3)            ratio={ratio_pca:.1f}x  RMSE={rmse_pca:.2f}C")

# Autoencoder
class AEComp(nn.Module):
    def __init__(self,w=32,b=3):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(w,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU(),nn.Linear(8,b))
        self.dec=nn.Sequential(nn.Linear(b,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU(),nn.Linear(16,w))
    def forward(self,x): return self.dec(self.enc(x))

ae_comp  = AEComp(WINDOW_COMP, BOTTLENECK)
opt_ae   = torch.optim.Adam(ae_comp.parameters(), lr=LR)
crit_ae  = nn.MSELoss()
print(f"  Training Autoencoder ...")
for ep in range(1, EPOCHS+1):
    ae_comp.train(); opt_ae.zero_grad()
    loss = crit_ae(ae_comp(Xt_comp), Xt_comp)
    loss.backward(); opt_ae.step()
    if ep % 10 == 0: print(f"    AE Epoch {ep}/{EPOCHS}  loss={loss.item():.5f}")
ae_comp.eval()
with torch.no_grad():
    X_ae_n = ae_comp(Xt_comp).numpy()
X_ae_o   = sc_comp.inverse_transform(X_ae_n)
T_ae     = wins_to_sig(X_ae_o, n)
rmse_ae  = float(np.sqrt(mean_squared_error(T_cal_input, T_ae)))
ratio_ae = WINDOW_COMP/BOTTLENECK
results.append({"Stage":"Compress","Method":"Autoencoder","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_ae,2),"Ratio":round(ratio_ae,1)})
print(f"  Autoencoder          ratio={ratio_ae:.1f}x  RMSE={rmse_ae:.2f}C")

# VAE
class VAEComp(nn.Module):
    def __init__(self,w=32,b=3):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(w,16),nn.ReLU(),nn.Linear(16,8),nn.ReLU())
        self.mu=nn.Linear(8,b); self.lv=nn.Linear(8,b)
        self.dec=nn.Sequential(nn.Linear(b,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU(),nn.Linear(16,w))
    def forward(self,x):
        h=self.enc(x); mu=self.mu(h); lv=self.lv(h)
        z=mu+torch.exp(0.5*lv)*torch.randn_like(mu)
        return self.dec(z),mu,lv

vae_comp = VAEComp(WINDOW_COMP, BOTTLENECK)
opt_vae  = torch.optim.Adam(vae_comp.parameters(), lr=LR)
print(f"  Training VAE ...")
for ep in range(1, EPOCHS+1):
    vae_comp.train(); opt_vae.zero_grad()
    recon,mu,lv = vae_comp(Xt_comp)
    loss = nn.MSELoss()(recon,Xt_comp) + 0.001*(-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp()))
    loss.backward(); opt_vae.step()
    if ep % 10 == 0: print(f"    VAE Epoch {ep}/{EPOCHS}  loss={loss.item():.5f}")
vae_comp.eval()
with torch.no_grad():
    X_vae_n,_,_ = vae_comp(Xt_comp)
    X_vae_n = X_vae_n.numpy()
X_vae_o  = sc_comp.inverse_transform(X_vae_n)
T_vae    = wins_to_sig(X_vae_o, n)
rmse_vae = float(np.sqrt(mean_squared_error(T_cal_input, T_vae)))
results.append({"Stage":"Compress","Method":"VAE","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_vae,2),"Ratio":round(WINDOW_COMP/BOTTLENECK,1)})
print(f"  VAE                  ratio={WINDOW_COMP/BOTTLENECK:.1f}x  RMSE={rmse_vae:.2f}C")

# Deep Autoencoder
class DeepAE(nn.Module):
    def __init__(self,w=32,b=3):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(w,24),nn.ReLU(),nn.Linear(24,16),nn.ReLU(),
                               nn.Linear(16,8),nn.ReLU(),nn.Linear(8,b))
        self.dec=nn.Sequential(nn.Linear(b,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU(),
                               nn.Linear(16,24),nn.ReLU(),nn.Linear(24,w))
    def forward(self,x): return self.dec(self.enc(x))

dae_comp = DeepAE(WINDOW_COMP, BOTTLENECK)
opt_dae  = torch.optim.Adam(dae_comp.parameters(), lr=LR)
print(f"  Training Deep Autoencoder ...")
for ep in range(1, EPOCHS+1):
    dae_comp.train(); opt_dae.zero_grad()
    loss = nn.MSELoss()(dae_comp(Xt_comp), Xt_comp)
    loss.backward(); opt_dae.step()
    if ep % 10 == 0: print(f"    DeepAE Epoch {ep}/{EPOCHS}  loss={loss.item():.5f}")
dae_comp.eval()
with torch.no_grad():
    X_dae_n = dae_comp(Xt_comp).numpy()
X_dae_o  = sc_comp.inverse_transform(X_dae_n)
T_dae    = wins_to_sig(X_dae_o, n)
rmse_dae = float(np.sqrt(mean_squared_error(T_cal_input, T_dae)))
results.append({"Stage":"Compress","Method":"Deep Autoencoder","Type":"ML",
                "Noise_C":"N/A","Reduction_%":"N/A","RMSE_C":round(rmse_dae,2),"Ratio":round(WINDOW_COMP/BOTTLENECK,1)})
print(f"  Deep Autoencoder     ratio={WINDOW_COMP/BOTTLENECK:.1f}x  RMSE={rmse_dae:.2f}C")


# =============================================================================
# SAVE FULL SUMMARY CSV
# =============================================================================
df_summary = pd.DataFrame(results)
df_summary.to_csv("full_summary.csv", index=False)
print()
print("=" * 65)
print("FULL SUMMARY saved --> full_summary.csv")
print("=" * 65)
print(df_summary.to_string(index=False))


# =============================================================================
# VISUALISATION
# =============================================================================
print()
print("Generating plots ...")

# ── Plot 1: Denoising ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("ATP-1 Denoising -- All Classical + ML Methods\nNIST Layer01 IN625",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(time_s, T_raw,    color="lightcoral", lw=0.5, alpha=0.5, label="Raw")
ax.plot(time_s, T_ma,     color="#3498db",    lw=0.9, label="Moving Avg")
ax.plot(time_s, T_med,    color="#2ecc71",    lw=0.9, label="Median")
ax.plot(time_s, T_sg,     color="#9b59b6",    lw=0.9, ls="--", label="Savitzky-Golay")
ax.plot(time_s, T_gauss,  color="#f39c12",    lw=0.9, label="Gaussian")
ax.plot(time_s, T_kalman, color="#1abc9c",    lw=0.9, ls=":", label="Kalman")
for name, c, ls in [("CNN","darkorange","-"),("LSTM","brown","--"),
                     ("AE Denoiser","purple","-."),("BiLSTM","red",":")]:
    ax.plot(time_s, T_den_ml[name], color=c, lw=0.9, ls=ls, label=name)
ax.set_title("All Methods Signal"); ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

ax = axes[1]
all_den = [("Raw Baseline",noise_raw,"Baseline"),
           ("Moving Avg",noise_level(T_ma),"Classical"),
           ("Median",noise_level(T_med),"Classical"),
           ("Savitzky-Golay",noise_level(T_sg),"Classical"),
           ("Gaussian",noise_level(T_gauss),"Classical"),
           ("Kalman",noise_level(T_kalman),"Classical"),
           ("CNN",noise_level(T_den_ml["CNN"]),"ML"),
           ("LSTM",noise_level(T_den_ml["LSTM"]),"ML"),
           ("AE Denoiser",noise_level(T_den_ml["AE Denoiser"]),"ML"),
           ("BiLSTM",noise_level(T_den_ml["BiLSTM"]),"ML")]
names_d = [x[0] for x in all_den]
vals_d  = [x[1] for x in all_den]
cols_d  = ["#e74c3c" if x[2]=="Baseline" else "#3498db" if x[2]=="Classical"
           else "#e67e22" for x in all_den]
bars = ax.bar(names_d, vals_d, color=cols_d, alpha=0.85)
for bar, val in zip(bars, vals_d):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_title("Noise Level Comparison (lower=better)")
ax.set_ylabel("Noise std dev (C)"); ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("full_comparison_denoise.png", dpi=150, bbox_inches="tight")
print("  Saved --> full_comparison_denoise.png")
plt.show()

# ── Plot 2: Calibration ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("ATP-2 Calibration -- All Classical + ML Methods\nNIST Layer01 IN625",
             fontsize=13, fontweight="bold")

t_te = time_s[WINDOW_CAL:][sp_c:]
ax   = axes[0]
ax.plot(t_te, y_te,  color="black",      lw=1.2, label="TC reference")
ax.plot(t_te, y_rf,  color="darkorange", lw=0.9, label=f"RF RMSE={rmse_rf:.1f}C")
ax.plot(t_te, y_mlp, color="green",      lw=0.9, ls="--", label=f"MLP RMSE={rmse_mlp:.1f}C")
ax.plot(t_te, y_gb,  color="purple",     lw=0.9, ls="-.", label=f"GradBoost RMSE={rmse_gb:.1f}C")
ax.plot(t_te, y_svr, color="brown",      lw=0.9, ls=":", label=f"SVR RMSE={rmse_svr:.1f}C")
ax.set_title("ML Methods vs TC Reference (test set)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
all_cal = [("Raw Baseline",rmse_raw_cal,"Baseline"),
           ("Mean Offset",rmse_off,"Classical"),
           ("Linear Regr.",rmse_lin,"Classical"),
           ("Polynomial",rmse_poly,"Classical"),
           ("Piecewise",rmse_pw,"Classical"),
           ("Random Forest",rmse_rf,"ML"),
           ("MLP Net",rmse_mlp,"ML"),
           ("GradBoost",rmse_gb,"ML"),
           ("SVR RBF",rmse_svr,"ML")]
names_c = [x[0] for x in all_cal]
vals_c  = [x[1] for x in all_cal]
cols_c  = ["#e74c3c" if x[2]=="Baseline" else "#3498db" if x[2]=="Classical"
           else "#e67e22" for x in all_cal]
bars = ax.bar(names_c, vals_c, color=cols_c, alpha=0.85)
for bar, val in zip(bars, vals_c):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_title("RMSE Comparison (lower=better)")
ax.set_ylabel("RMSE vs TC reference (C)"); ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=30, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("full_comparison_calibrate.png", dpi=150, bbox_inches="tight")
print("  Saved --> full_comparison_calibrate.png")
plt.show()

# ── Plot 3: Compression ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("ATP-3 Compression -- All Classical + ML Methods\nNIST Layer01 IN625",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(time_s, T_cal_input, color="black",      lw=1.2, label="Calibrated")
ax.plot(time_s, T_wav,       color="#3498db",    lw=0.9, label=f"Wavelet {ratio_wav:.1f}x")
ax.plot(time_s, T_ds,        color="#2ecc71",    lw=0.9, ls="--", label="Downsamp 2x")
ax.plot(time_s, T_pca,       color="darkorange", lw=0.9, label=f"PCA {ratio_pca:.1f}x")
ax.plot(time_s, T_ae,        color="green",      lw=0.9, ls="-.", label=f"AE {ratio_ae:.1f}x")
ax.plot(time_s, T_vae, color="purple", lw=0.9, ls=":", label=f"VAE {WINDOW_COMP/BOTTLENECK:.1f}x")
ax.plot(time_s, T_dae,       color="brown",      lw=0.9, ls="--", label=f"DeepAE {ratio_ae:.1f}x")
ax.set_title("Compressed Signals vs Original")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[1]
all_comp = [("Raw\nBaseline",0.0,1.0,"Baseline"),
            ("Delta\nEnc",rmse_del,ratio_del,"Classical"),
            ("Downsamp\n2x",rmse_ds,2.0,"Classical"),
            ("SVD\nr=10",rmse_svd,ratio_svd,"Classical"),
            ("Wavelet\n10%",rmse_wav,ratio_wav,"Classical"),
            ("PCA\nn=3",rmse_pca,ratio_pca,"ML"),
            ("AE",rmse_ae,ratio_ae,"ML"),
            ("VAE",rmse_vae,WINDOW_COMP/BOTTLENECK,"ML"),
            ("DeepAE",rmse_dae,WINDOW_COMP/BOTTLENECK,"ML")]
cols_comp = ["#e74c3c" if x[3]=="Baseline" else "#3498db" if x[3]=="Classical"
             else "#e67e22" for x in all_comp]
ax.scatter([x[2] for x in all_comp], [x[1] for x in all_comp],
           c=cols_comp, s=150, zorder=5, edgecolors="black", lw=0.8)
for x in all_comp:
    ax.annotate(x[0], (x[2], x[1]), textcoords="offset points",
                xytext=(5,5), fontsize=8)
ax.axvline(x=4.0, color="green", ls="--", lw=1.5, label="4x target")
ax.set_title("RMSE vs Ratio (bottom-right = best)")
ax.set_xlabel("Compression ratio"); ax.set_ylabel("RMSE (C)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("full_comparison_compress.png", dpi=150, bbox_inches="tight")
print("  Saved --> full_comparison_compress.png")
plt.show()

print()
print("=" * 65)
print("ALL DONE!")
print("=" * 65)
print("  Outputs:")
print("    full_summary.csv")
print("    full_comparison_denoise.png")
print("    full_comparison_calibrate.png")
print("    full_comparison_compress.png")
print()
print("  BEST METHODS SUMMARY:")
best_den  = min([(noise_level(T_den_ml[n]),n) for n in T_den_ml])[1]
best_cal  = min([(rmse_rf,"RF"),(rmse_mlp,"MLP"),(rmse_gb,"GradBoost"),(rmse_svr,"SVR")])[1]
print(f"  Denoising   : {best_den} (best ML)  |  Median (best classical)")
print(f"  Calibration : {best_cal} (best ML)  |  Linear {rmse_lin:.1f}C (classical)")
print(f"  Compression : Wavelet 10% {ratio_wav:.1f}x {rmse_wav:.1f}C (best classical)")
print(f"                PCA/AE {ratio_pca:.1f}x (best ML ratio)")
print("=" * 65)
