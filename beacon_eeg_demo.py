"""
Beacon Biosignals-style Algorithm Engineer Demo
================================================
Synthetic multi-channel EEG sleep staging pipeline covering:
- Digital signal processing (filtering, PSD, spectrograms)
- Feature extraction (relative band power)
- Lightweight PyTorch CNN (EEGNet-inspired) for sleep stage classification
- Training, evaluation, and rich visualizations

Sleep stages simulated: Wake, N1, N2, N3, REM
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report
import os

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# 1. Synthetic EEG generation (characteristic frequency bands per stage)
# ---------------------------------------------------------------------------
STAGES = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_BANDS = {
    "Wake": {"alpha": 10.0, "beta": 20.0, "amp": 1.2},
    "N1":   {"theta": 6.0, "amp": 0.8},
    "N2":   {"theta": 6.0, "sigma": 13.0, "amp": 1.0},   # spindles
    "N3":   {"delta": 1.5, "amp": 2.0},                  # slow waves
    "REM":  {"theta": 6.5, "beta": 25.0, "amp": 0.9},
}

FS = 100          # sampling rate (Hz)
EPOCH_SEC = 30
N_SAMPLES = FS * EPOCH_SEC
N_CHANNELS = 4
N_EPOCHS_PER_STAGE = 80


def generate_stage_epoch(stage: str) -> np.ndarray:
    """Generate one multi-channel epoch with stage-specific spectral content + noise."""
    t = np.arange(N_SAMPLES) / FS
    data = np.zeros((N_CHANNELS, N_SAMPLES))
    params = STAGE_BANDS[stage]

    for ch in range(N_CHANNELS):
        sig = np.zeros(N_SAMPLES)
        # Base pink-ish noise
        noise = np.random.randn(N_SAMPLES)
        b, a = signal.butter(2, 0.5 / (FS / 2), btype="high")
        noise = signal.filtfilt(b, a, noise) * 0.3

        # Stage-specific oscillations
        if "delta" in params:
            sig += params["amp"] * np.sin(2 * np.pi * params["delta"] * t + np.random.rand() * 2 * np.pi)
        if "theta" in params:
            sig += params["amp"] * 0.8 * np.sin(2 * np.pi * params["theta"] * t + np.random.rand() * 2 * np.pi)
        if "alpha" in params:
            sig += params["amp"] * np.sin(2 * np.pi * params["alpha"] * t + np.random.rand() * 2 * np.pi)
        if "sigma" in params:  # spindles – bursty
            spindle = np.sin(2 * np.pi * params["sigma"] * t)
            envelope = np.exp(-0.5 * ((t - 15) / 2) ** 2)  # centered burst
            sig += params["amp"] * 0.6 * spindle * envelope
        if "beta" in params:
            sig += params["amp"] * 0.4 * np.sin(2 * np.pi * params["beta"] * t + np.random.rand() * 2 * np.pi)

        # Channel-specific slight variation + noise
        data[ch] = sig * (0.8 + 0.4 * np.random.rand()) + noise

    return data


print("Generating synthetic EEG dataset...")
X_list, y_list = [], []
for stage_idx, stage in enumerate(STAGES):
    for _ in range(N_EPOCHS_PER_STAGE):
        X_list.append(generate_stage_epoch(stage))
        y_list.append(stage_idx)

X = np.stack(X_list)          # (n_epochs, n_channels, n_samples)
y = np.array(y_list)
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ---------------------------------------------------------------------------
# 2. DSP visualizations
# ---------------------------------------------------------------------------
os.makedirs("/workspace/artifacts/plots", exist_ok=True)
sns.set_style("whitegrid")

# Raw example epochs
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
for i, stage in enumerate(STAGES):
    epoch = X[y == i][0, 0]  # first channel of first example
    axes[i].plot(np.arange(N_SAMPLES) / FS, epoch, color=sns.color_palette()[i])
    axes[i].set_ylabel(stage)
    axes[i].set_title(f"Example {stage} epoch (ch 0)")
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/01_raw_epochs.png", dpi=150)
plt.close()

# Band-pass filter example (0.5–40 Hz)
def bandpass(data, low=0.5, high=40, fs=FS):
    b, a = signal.butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, data, axis=-1)

X_filt = bandpass(X)

# Bandpass-filtered example epochs
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
for i, stage in enumerate(STAGES):
    epoch = X_filt[y == i][0, 0]  # first channel of first example
    axes[i].plot(np.arange(N_SAMPLES) / FS, epoch, color=sns.color_palette()[i])
    axes[i].set_ylabel(stage)
    axes[i].set_title(f"Bandpass-filtered {stage} epoch (0.5–40 Hz, ch 0)")
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/02_bandpass_epochs.png", dpi=150)
plt.close()

# PSD comparison — one panel per stage so peaks are easy to see
fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
freqs = np.fft.rfftfreq(N_SAMPLES, 1 / FS)
for i, stage in enumerate(STAGES):
    epoch = X_filt[y == i][0, 0]
    psd = np.abs(np.fft.rfft(epoch)) ** 2
    axes[i].semilogy(freqs, psd, color=sns.color_palette()[i])
    axes[i].set_ylabel(stage)
    axes[i].set_title(f"PSD — {stage}")
    axes[i].set_xlim(0, 40)
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel("Frequency (Hz)")
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/03_psd_by_stage.png", dpi=150)
plt.close()

# Spectrogram of one long concatenated night (simple demo)
long_signal = np.concatenate([X_filt[y == i][0, 0] for i in range(5)])
f, t_spec, Sxx = signal.spectrogram(long_signal, FS, nperseg=256, noverlap=128)
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
ax.set_ylim(0, 40)
ax.set_title("Spectrogram across stages (synthetic night fragment)")
plt.colorbar(im, ax=ax, label="Power (dB)")
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/04_spectrogram.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Feature extraction: relative band power
# ---------------------------------------------------------------------------
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "sigma": (12, 15),
    "beta": (15, 30),
}

def relative_band_power(epoch, fs=FS):
    """Return relative power in each band (mean across channels)."""
    freqs, psd = signal.welch(epoch, fs=fs, nperseg=min(256, epoch.shape[-1]), axis=-1)
    total = np.trapezoid(psd, freqs, axis=-1) + 1e-12
    features = []
    for low, high in BANDS.values():
        idx = (freqs >= low) & (freqs < high)
        band_power = np.trapezoid(psd[:, idx], freqs[idx], axis=-1)
        features.append((band_power / total).mean())  # average channels
    return np.array(features)

print("Extracting band-power features...")
X_feat = np.array([relative_band_power(ep) for ep in X_filt])
print(f"Feature matrix shape: {X_feat.shape}")

# Feature visualization
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    np.array([X_feat[y == i].mean(0) for i in range(5)]),
    xticklabels=list(BANDS.keys()),
    yticklabels=STAGES,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    ax=ax,
)
ax.set_title("Mean Relative Band Power by Sleep Stage")
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/05_band_power_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. PyTorch model (compact EEGNet-inspired CNN)
# ---------------------------------------------------------------------------
class EEGNetSleep(nn.Module):
    """Lightweight EEGNet-style model for sleep staging."""
    def __init__(self, n_channels=N_CHANNELS, n_samples=N_SAMPLES, n_classes=5):
        super().__init__()
        # Temporal convolution
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16),
        )
        # Depthwise spatial
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
        )
        # Separable
        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
        )
        # Classifier
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out = self.forward_features(dummy)
            n_flat = out.view(1, -1).shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def forward_features(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Prepare tensors: (batch, 1, channels, samples)
X_tensor = torch.tensor(X_filt.copy(), dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.long)

# Train/val split
n = len(X_tensor)
idx = np.random.permutation(n)
split = int(0.8 * n)
train_ds = TensorDataset(X_tensor[idx[:split]], y_tensor[idx[:split]])
val_ds = TensorDataset(X_tensor[idx[split:]], y_tensor[idx[split:]])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGNetSleep().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print(f"Training on {device}...")
train_losses, val_losses, val_accs = [], [], []

for epoch in range(15):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    train_losses.append(running / len(train_loader))

    model.eval()
    vloss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            vloss += criterion(logits, yb).item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    val_losses.append(vloss / len(val_loader))
    val_accs.append(correct / total)
    print(f"Epoch {epoch+1:02d} | train_loss={train_losses[-1]:.3f} | val_loss={val_losses[-1]:.3f} | val_acc={val_accs[-1]:.3f}")

# ---------------------------------------------------------------------------
# 5. Evaluation plots
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label="Train")
ax1.plot(val_losses, label="Val")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Curves")
ax1.legend()
ax2.plot(val_accs, color="green")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Validation Accuracy")
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/06_training_curves.png", dpi=150)
plt.close()

# Confusion matrix
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        preds = model(xb.to(device)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=STAGES, yticklabels=STAGES, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix (Validation)")
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/07_confusion_matrix.png", dpi=150)
plt.close()

print("\nClassification Report:")
print(classification_report(all_true, all_preds, target_names=STAGES))

# Simulated hypnogram (sequential stages for visualization)
hypno_true = np.array([0, 0, 1, 2, 2, 3, 3, 3, 2, 4, 4, 2, 1, 0])
# Fake predictions with a couple of errors for demo
hypno_pred = hypno_true.copy()
hypno_pred[5] = 2
hypno_pred[10] = 2

fig, ax = plt.subplots(figsize=(10, 3))
ax.step(range(len(hypno_true)), hypno_true, where="post", label="True", linewidth=2)
ax.step(range(len(hypno_pred)), hypno_pred, where="post", label="Predicted", linestyle="--", linewidth=2)
ax.set_yticks(range(5))
ax.set_yticklabels(STAGES)
ax.set_xlabel("Epoch")
ax.set_title("Example Hypnogram Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("/workspace/artifacts/plots/08_hypnogram.png", dpi=150)
plt.close()

print("\nAll plots saved to /workspace/artifacts/plots/")
print("Demo complete. This covers DSP, feature engineering, PyTorch modeling,")
print("evaluation metrics, and visualization — core skills for the Beacon role.")
