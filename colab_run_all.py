"""
FinJEPA — Colab Quick-Start Script
====================================
Copy-paste this ENTIRE file into a single Colab cell.
It includes everything inline — no file uploads needed.

NOTE: 
    - This takes ~15-20 min to run end-to-end on a Colab T4 GPU
    - If you want faster iteration, reduce ts2vec_epochs and patchtst_epochs to 50
    
Steps:
    1. Open Google Colab
    2. Runtime → Change runtime type → GPU (T4)
    3. Paste this script, run it
    4. Download results from /content/results/
"""

# ─────────────────────────────────────────────
# Cell 1: Install dependencies
# ─────────────────────────────────────────────
# !pip install -q yfinance hmmlearn umap-learn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import yfinance as yf
    HAS_YF = True
except:
    HAS_YF = False
    print("Install yfinance: !pip install yfinance")

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except:
    HAS_HMM = False
    print("Install hmmlearn: !pip install hmmlearn")

try:
    from umap import UMAP
    HAS_UMAP = True
except:
    HAS_UMAP = False
    print("Install umap: !pip install umap-learn")

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════
# SECTION 1: DATA PIPELINE
# ══════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("SECTION 1: DATA PIPELINE")
print("█"*60)

# Download S&P 500
cache_path = RESULTS_DIR / "sp500_raw.csv"
if cache_path.exists():
    print(f"Loading cached data...")
    raw_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
else:
    print("Downloading S&P 500 data (2000-2024)...")
    raw_df = yf.download("^GSPC", start="2000-01-01", end="2024-12-31", auto_adjust=True)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    raw_df.to_csv(cache_path)

print(f"Data: {len(raw_df)} rows, {raw_df.index[0].date()} to {raw_df.index[-1].date()}")

# Log returns
close = raw_df['Close'].values.astype(np.float64)
log_returns = np.log(close[1:] / close[:-1])
dates = raw_df.index[1:]
returns_df = pd.DataFrame({'close': close[1:], 'log_return': log_returns}, index=dates)

# Rolling z-score (252-day causal window)
ROLLING_WINDOW = 252
series = pd.Series(log_returns)
rolling_mean = series.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
rolling_std = series.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).std().replace(0, np.nan)
z_returns = ((series - rolling_mean) / rolling_std).values
returns_df['z_return'] = z_returns

# Patches (20-day non-overlapping)
PATCH_SIZE = 20
valid_mask = ~np.isnan(returns_df['z_return'].values)
valid_data = returns_df[valid_mask]
n_days = len(valid_data)
n_patches = n_days // PATCH_SIZE
truncated = n_patches * PATCH_SIZE

z_vals = valid_data['z_return'].values[:truncated]
raw_vals = valid_data['log_return'].values[:truncated]
patch_dates_idx = valid_data.index[:truncated]

patches = z_vals.reshape(n_patches, PATCH_SIZE)
raw_patches = raw_vals.reshape(n_patches, PATCH_SIZE)

patch_dates = []
for i in range(n_patches):
    s = i * PATCH_SIZE
    e = (i + 1) * PATCH_SIZE - 1
    patch_dates.append((patch_dates_idx[s], patch_dates_idx[e]))

print(f"Total patches: {n_patches}, shape: {patches.shape}")

# Temporal split
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_START = pd.Timestamp("2020-01-01")
VAL_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

train_idx, val_idx, test_idx = [], [], []
for i, (s, e) in enumerate(patch_dates):
    if e <= TRAIN_END: train_idx.append(i)
    elif s >= VAL_START and e <= VAL_END: val_idx.append(i)
    elif s >= TEST_START: test_idx.append(i)

train_patches = patches[train_idx]
val_patches = patches[val_idx]
test_patches = patches[test_idx]

print(f"Split: Train={len(train_patches)}, Val={len(val_patches)}, Test={len(test_patches)}")

# Daily returns split for HMM
valid_returns = returns_df.dropna(subset=['z_return'])
train_daily = valid_returns[valid_returns.index <= TRAIN_END]
val_daily = valid_returns[(valid_returns.index >= VAL_START) & (valid_returns.index <= VAL_END)]
test_daily = valid_returns[valid_returns.index >= TEST_START]

print(f"Daily: Train={len(train_daily)}, Val={len(val_daily)}, Test={len(test_daily)}")


# ══════════════════════════════════════════════════════════════
# SECTION 2: HMM GROUND TRUTH (Track A)
# ══════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("SECTION 2: HMM GROUND TRUTH (Track A)")
print("█"*60)

N_STATES = 3

hmm = GaussianHMM(n_components=N_STATES, covariance_type="full", n_iter=200, random_state=42, tol=1e-4)
train_rets = train_daily['log_return'].values
hmm.fit(train_rets.reshape(-1, 1))

# Sort states by mean return
means = hmm.means_.flatten()
state_order = np.argsort(means)

state_names_list = ['Bear', 'Sideways', 'Bull']
print("\nHMM States (sorted by mean return):")
for new_idx, old_idx in enumerate(state_order):
    m = hmm.means_[old_idx, 0]
    s = np.sqrt(hmm.covars_[old_idx, 0, 0])
    print(f"  {state_names_list[new_idx]:>10}: mean={m:+.6f}, std={s:.6f}, "
          f"ann_mean={m*252:.2%}, ann_vol={s*np.sqrt(252):.2%}")

# Create mapping
reverse_map = np.zeros(N_STATES, dtype=int)
for new_idx, old_idx in enumerate(state_order):
    reverse_map[old_idx] = new_idx

def predict_regime(returns_arr):
    raw = hmm.predict(returns_arr.reshape(-1, 1))
    return reverse_map[raw]

def labels_to_patches_fn(daily_labels, ps=PATCH_SIZE):
    n = len(daily_labels)
    np_ = n // ps
    return stats.mode(daily_labels[:np_*ps].reshape(np_, ps), axis=1, keepdims=False).mode.astype(int)

# Label everything
hmm_daily = {}
hmm_patch = {}
for split_name, split_daily in [('train', train_daily), ('val', val_daily), ('test', test_daily)]:
    rets = split_daily['log_return'].values
    daily_labels = predict_regime(rets)
    hmm_daily[split_name] = daily_labels
    hmm_patch[split_name] = labels_to_patches_fn(daily_labels)
    
    unique, counts = np.unique(daily_labels, return_counts=True)
    print(f"\n{split_name.upper()} daily regime distribution:")
    for st, cnt in zip(unique, counts):
        print(f"  {state_names_list[st]:>10}: {cnt:5d} ({cnt/len(daily_labels)*100:.1f}%)")

# Export labels for Mehul
labels_dir = RESULTS_DIR / "labels"
labels_dir.mkdir(exist_ok=True)
for split_name in ['train', 'val', 'test']:
    np.save(labels_dir / f"hmm_daily_labels_{split_name}.npy", hmm_daily[split_name])
    np.save(labels_dir / f"hmm_patch_labels_{split_name}.npy", hmm_patch[split_name])
print(f"\n✅ HMM labels exported to {labels_dir}/")
print(f"   → hmm_patch_labels_val.npy  ({len(hmm_patch['val'])} patches) ← for Mehul")
print(f"   → hmm_patch_labels_test.npy ({len(hmm_patch['test'])} patches) ← for Mehul")

# Plot regimes
fig, axes = plt.subplots(2, 1, figsize=(18, 8))
colors_map = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}

# Price + regimes
all_daily_list = []
for sn, sd, dl in [('train', train_daily, hmm_daily['train']), 
                    ('val', val_daily, hmm_daily['val']), 
                    ('test', test_daily, hmm_daily['test'])]:
    df_tmp = sd.copy()
    df_tmp['regime'] = dl
    all_daily_list.append(df_tmp)
combined = pd.concat(all_daily_list)

ax = axes[0]
ax.set_title('S&P 500 with HMM Regime Labels', fontsize=14, fontweight='bold')
for r in [0, 1, 2]:
    mask = combined['regime'] == r
    ax.scatter(combined.index[mask], combined['close'][mask], c=colors_map[r], s=1, alpha=0.7, label=state_names_list[r])
ax.axvline(TRAIN_END, color='gray', ls='--', alpha=0.5)
ax.axvline(VAL_END, color='gray', ls='--', alpha=0.5)
ax.legend(markerscale=10)
ax.set_ylabel('Price')
ax.grid(True, alpha=0.2)

ax = axes[1]
ax.set_title('Regime Timeline', fontsize=12)
for r in [0, 1, 2]:
    mask = combined['regime'] == r
    ax.scatter(combined.index[mask], [r]*mask.sum(), c=colors_map[r], s=2, alpha=0.8)
ax.set_yticks([0,1,2])
ax.set_yticklabels(['Bear','Sideways','Bull'])
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "regime_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Regime plot saved")


# ══════════════════════════════════════════════════════════════
# SECTION 3: SUPERVISED BASELINE
# ══════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("SECTION 3: SUPERVISED BASELINE")
print("█"*60)

CONTEXT_LEN = 12

class RegimeDataset(Dataset):
    def __init__(self, patches, labels, context_len=CONTEXT_LEN):
        self.patches = torch.FloatTensor(patches)
        self.labels = torch.LongTensor(labels)
        self.context_len = context_len
        self.n_samples = max(0, len(patches) - context_len + 1)
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        ctx = self.patches[idx:idx+self.context_len]
        lbl = self.labels[idx+self.context_len-1]
        return ctx, lbl

class SupervisedBaseline(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, d_model=384, n_heads=6, n_layers=6, n_classes=N_STATES, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, dropout, 'gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, n_classes))
    
    def get_representations(self, x):
        b, n, _ = x.shape
        x = self.patch_embed(x) + self.pos_embed[:,:n,:]
        x = self.norm(self.encoder(x))
        return x.mean(dim=1)
    
    def forward(self, x):
        return self.classifier(self.get_representations(x))

# Create datasets
train_ds = RegimeDataset(train_patches, hmm_patch['train'])
val_ds = RegimeDataset(val_patches, hmm_patch['val'])
test_ds = RegimeDataset(test_patches, hmm_patch['test'])
print(f"Supervised datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# Train
sup_model = SupervisedBaseline().to(DEVICE)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# Class weights (handle missing classes)
all_labels = [train_ds[i][1].item() for i in range(len(train_ds))]
unique_classes, counts = np.unique(all_labels, return_counts=True)
weight_arr = np.ones(N_STATES, dtype=np.float32)
for cls, cnt in zip(unique_classes, counts):
    weight_arr[cls] = 1.0 / cnt
weight_arr = weight_arr / weight_arr.sum() * N_STATES
weights = torch.FloatTensor(weight_arr).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(sup_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_f1, best_state, patience = 0, None, 0
for epoch in range(100):
    sup_model.train()
    losses = []
    for p, l in train_loader:
        p, l = p.to(DEVICE), l.to(DEVICE)
        loss = criterion(sup_model(p), l)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(sup_model.parameters(), 1.0)
        optimizer.step(); losses.append(loss.item())
    scheduler.step()
    
    sup_model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for p, l in val_loader:
            p = p.to(DEVICE)
            preds_list.extend(sup_model(p).argmax(1).cpu().numpy())
            labels_list.extend(l.numpy())
    vf1 = f1_score(labels_list, preds_list, average='macro')
    
    if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1:3d} | Loss: {np.mean(losses):.4f} | Val F1: {vf1:.4f}")
    if vf1 > best_f1:
        best_f1 = vf1; best_state = {k: v.cpu().clone() for k, v in sup_model.state_dict().items()}; patience = 0
    else:
        patience += 1
        if patience >= 15: print(f"  Early stop at epoch {epoch+1}"); break

if best_state: sup_model.load_state_dict(best_state); sup_model.to(DEVICE)
print(f"✅ Supervised baseline trained (best val F1: {best_f1:.4f})")

# Extract representations
def extract_reprs(model, dataset):
    model.eval(); loader = DataLoader(dataset, batch_size=64, shuffle=False)
    reprs, lbls = [], []
    with torch.no_grad():
        for p, l in loader:
            reprs.append(model.get_representations(p.to(DEVICE)).cpu().numpy())
            lbls.append(l.numpy())
    return np.concatenate(reprs), np.concatenate(lbls)

sup_val_reprs, sup_val_labels = extract_reprs(sup_model, val_ds)
sup_test_reprs, sup_test_labels = extract_reprs(sup_model, test_ds)
print(f"Supervised reprs: val={sup_val_reprs.shape}, test={sup_test_reprs.shape}")

# ══════════════════════════════════════════════════════════════
# SECTION 4: TS2Vec BASELINE (Track D)
# ══════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("SECTION 4: TS2Vec BASELINE (Track D)")
print("█"*60)

class DilatedConvBlock(nn.Module):
    def __init__(self, ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size-1)*dilation
        self.conv = nn.Conv1d(ch, ch, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(ch)
        self.act = nn.GELU()
        self.trim = pad
    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        if self.trim > 0: out = out[:,:,:-self.trim]
        return out + x

class TSEncoder(nn.Module):
    def __init__(self, input_dims=1, hidden=64, output_dims=384, depth=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dims, hidden)
        self.layers = nn.ModuleList([DilatedConvBlock(hidden, dilation=2**i) for i in range(depth)])
        self.output_proj = nn.Linear(hidden, output_dims)
    def forward(self, x, mask=None):
        x = self.input_proj(x)
        if mask is not None: x = x * mask.unsqueeze(-1)
        x = x.transpose(1,2)
        for layer in self.layers: x = layer(x)
        return self.output_proj(x.transpose(1,2))

# Prepare data — sliding windows of 12 patches flattened
sequences = np.array([train_patches[i:i+CONTEXT_LEN].flatten() for i in range(len(train_patches)-CONTEXT_LEN+1)])
print(f"TS2Vec training: {sequences.shape}")

ts_encoder = TSEncoder(input_dims=1, hidden=64, output_dims=384, depth=8).to(DEVICE)
ts_optimizer = torch.optim.AdamW(ts_encoder.parameters(), lr=1e-3, weight_decay=1e-4)
ts_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ts_optimizer, 200)

ts_data = torch.FloatTensor(sequences[..., np.newaxis])
ts_loader = DataLoader(TensorDataset(ts_data), batch_size=min(16, len(ts_data)//2), shuffle=True, drop_last=True)

TS2VEC_EPOCHS = 200
ts_encoder.train()
for epoch in range(TS2VEC_EPOCHS):
    losses = []
    for (bx,) in ts_loader:
        bx = bx.to(DEVICE)
        # Two augmented views (random crop + timestamp mask)
        cl = max(1, int(bx.size(1)*0.9))
        s1, s2 = np.random.randint(0, max(1,bx.size(1)-cl+1)), np.random.randint(0, max(1,bx.size(1)-cl+1))
        c1, c2 = bx[:,s1:s1+cl], bx[:,s2:s2+cl]
        ml = min(c1.size(1), c2.size(1)); c1, c2 = c1[:,:ml], c2[:,:ml]
        m1 = (torch.rand(c1.shape[0], ml, device=DEVICE) > 0.5).float()
        m2 = (torch.rand(c2.shape[0], ml, device=DEVICE) > 0.5).float()
        z1, z2 = ts_encoder(c1, m1), ts_encoder(c2, m2)
        # Simplified contrastive loss
        z1n, z2n = F.normalize(z1.mean(1), dim=1), F.normalize(z2.mean(1), dim=1)
        sim = torch.mm(z1n, z2n.t()) / 0.5
        labels_ts = torch.arange(len(z1n), device=DEVICE)
        loss = (F.cross_entropy(sim, labels_ts) + F.cross_entropy(sim.t(), labels_ts)) / 2
        ts_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ts_encoder.parameters(), 1.0)
        ts_optimizer.step(); losses.append(loss.item())
    ts_scheduler.step()
    if (epoch+1) % 40 == 0: print(f"  TS2Vec Epoch {epoch+1:3d} | Loss: {np.mean(losses):.4f}")

print("✅ TS2Vec trained")

# Extract representations
ts_encoder.eval()
def ts2vec_encode(patches_arr):
    seqs = np.array([patches_arr[i:i+CONTEXT_LEN].flatten() for i in range(len(patches_arr)-CONTEXT_LEN+1)])
    data = torch.FloatTensor(seqs[..., np.newaxis])
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=False)
    reprs = []
    with torch.no_grad():
        for (bx,) in loader:
            z = ts_encoder(bx.to(DEVICE))
            reprs.append(z[:, -PATCH_SIZE:, :].mean(1).cpu().numpy())
    return np.concatenate(reprs)

ts2vec_val_reprs = ts2vec_encode(val_patches)
ts2vec_test_reprs = ts2vec_encode(test_patches)
print(f"TS2Vec reprs: val={ts2vec_val_reprs.shape}, test={ts2vec_test_reprs.shape}")


# ══════════════════════════════════════════════════════════════
# SECTION 5: PatchTST BASELINE (Track E)
# ══════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("SECTION 5: PatchTST BASELINE (Track E)")
print("█"*60)

class PatchTSTEncoder(nn.Module):
    def __init__(self, patch_size=PATCH_SIZE, d_model=384, n_heads=6, n_layers=6, mask_ratio=0.4, dropout=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, dropout, 'gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.recon_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, patch_size))
    
    def forward_pretrain(self, x):
        b, n, _ = x.shape
        emb = self.patch_embed(x) + self.pos_embed[:,:n,:]
        # Random mask
        n_mask = max(1, int(n * self.mask_ratio))
        mask = torch.zeros(b, n, dtype=torch.bool, device=x.device)
        for i in range(b):
            idx = torch.randperm(n, device=x.device)[:n_mask]
            mask[i, idx] = True
        emb_masked = emb.clone()
        emb_masked[mask] = self.mask_token.squeeze()
        encoded = self.norm(self.encoder(emb_masked))
        pred = self.recon_head(encoded)
        loss = F.mse_loss(pred[mask], x[mask])
        return loss
    
    def forward(self, x):
        b, n, _ = x.shape
        emb = self.patch_embed(x) + self.pos_embed[:,:n,:]
        return self.norm(self.encoder(emb)).mean(dim=1)

# Prepare windows
pt_windows = np.array([train_patches[i:i+CONTEXT_LEN] for i in range(len(train_patches)-CONTEXT_LEN+1)])
print(f"PatchTST training: {pt_windows.shape}")

pt_model = PatchTSTEncoder().to(DEVICE)
pt_optimizer = torch.optim.AdamW(pt_model.parameters(), lr=1e-4, weight_decay=1e-4)
pt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pt_optimizer, 200)

pt_loader = DataLoader(TensorDataset(torch.FloatTensor(pt_windows)), batch_size=min(32, len(pt_windows)//2), shuffle=True, drop_last=True)

pt_model.train()
for epoch in range(200):
    losses = []
    for (bx,) in pt_loader:
        loss = pt_model.forward_pretrain(bx.to(DEVICE))
        pt_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(pt_model.parameters(), 1.0)
        pt_optimizer.step(); losses.append(loss.item())
    pt_scheduler.step()
    if (epoch+1) % 40 == 0: print(f"  PatchTST Epoch {epoch+1:3d} | Recon Loss: {np.mean(losses):.6f}")

print("✅ PatchTST trained")

# Extract representations
pt_model.eval()
def patchtst_encode(patches_arr):
    windows = np.array([patches_arr[i:i+CONTEXT_LEN] for i in range(len(patches_arr)-CONTEXT_LEN+1)])
    data = torch.FloatTensor(windows)
    loader = DataLoader(TensorDataset(data), batch_size=64, shuffle=False)
    reprs = []
    with torch.no_grad():
        for (bx,) in loader:
            reprs.append(pt_model(bx.to(DEVICE)).cpu().numpy())
    return np.concatenate(reprs)

pt_val_reprs = patchtst_encode(val_patches)
pt_test_reprs = patchtst_encode(test_patches)
print(f"PatchTST reprs: val={pt_val_reprs.shape}, test={pt_test_reprs.shape}")


# ══════════════════════════════════════════════════════════════
# SECTION 6: UNIFIED EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n" + "█"*60)
print("SECTION 6: UNIFIED EVALUATION")
print("█"*60)

def eval_model(name, val_r, test_r, val_l, test_l):
    """Evaluate with linear probe."""
    scaler = StandardScaler()
    vr = scaler.fit_transform(val_r)
    tr = scaler.transform(test_r)
    probe = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    probe.fit(vr, val_l)
    preds = probe.predict(tr)
    f1 = f1_score(test_l, preds, average='macro')
    acc = np.mean(preds == test_l)
    cm = confusion_matrix(test_l, preds)
    sil = None
    umap_emb = None
    if HAS_UMAP and len(tr) > 10:
        try:
            umap_emb = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(tr)-1)).fit_transform(tr)
            sil = silhouette_score(umap_emb, test_l)
        except: pass
    print(f"  {name:>15}: F1={f1:.4f}, Acc={acc:.4f}" + (f", Sil={sil:.4f}" if sil else ""))
    return {'model': name, 'f1': f1, 'acc': acc, 'cm': cm, 'sil': sil, 'umap': umap_emb}

# Align labels with representation lengths
all_results = []

# Random baseline
np.random.seed(42)
test_lab = hmm_patch['test']
rand_preds = np.random.randint(0, N_STATES, len(test_lab))
all_results.append({'model': 'Random', 'f1': f1_score(test_lab, rand_preds, average='macro'), 'acc': np.mean(rand_preds == test_lab), 'cm': None, 'sil': None, 'umap': None})
print(f"  {'Random':>15}: F1={all_results[-1]['f1']:.4f}")

# Supervised
all_results.append(eval_model('Supervised', sup_val_reprs, sup_test_reprs, sup_val_labels, sup_test_labels))

# TS2Vec (align labels — fewer outputs due to context windowing)
n_tv = len(ts2vec_val_reprs)
n_tt = len(ts2vec_test_reprs)
all_results.append(eval_model('TS2Vec', ts2vec_val_reprs, ts2vec_test_reprs, hmm_patch['val'][-n_tv:], hmm_patch['test'][-n_tt:]))

# PatchTST
n_pv = len(pt_val_reprs)
n_pt = len(pt_test_reprs)
all_results.append(eval_model('PatchTST', pt_val_reprs, pt_test_reprs, hmm_patch['val'][-n_pv:], hmm_patch['test'][-n_pt:]))

# Results table
print("\n" + "="*60)
print("FINAL RESULTS TABLE")
print("="*60)
results_df = pd.DataFrame([{
    'Model': r['model'],
    'Regime F1': f"{r['f1']:.4f}",
    'Silhouette': f"{r['sil']:.4f}" if r.get('sil') else '—',
    'Labels Used?': 'Yes' if r['model'] == 'Supervised' else 'No'
} for r in all_results])
print(results_df.to_string(index=False))
results_df.to_csv(RESULTS_DIR / "results_table.csv", index=False)

# UMAP plots
models_with_umap = [r for r in all_results if r.get('umap') is not None]
if models_with_umap:
    fig, axes = plt.subplots(1, len(models_with_umap), figsize=(6*len(models_with_umap), 5))
    if len(models_with_umap) == 1: axes = [axes]
    for ax, r in zip(axes, models_with_umap):
        emb = r['umap']
        n = len(emb)
        lab = hmm_patch['test'][-n:]
        for regime in range(N_STATES):
            m = lab == regime
            if m.any():
                ax.scatter(emb[m,0], emb[m,1], c=colors_map[regime], s=30, alpha=0.7, label=state_names_list[regime], edgecolors='white', linewidth=0.3)
        ax.set_title(f"{r['model']} (Sil: {r['sil']:.3f})" if r['sil'] else r['model'], fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "umap_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, len([r for r in all_results if r.get('cm') is not None]), 
                         figsize=(5*len([r for r in all_results if r.get('cm') is not None]), 4))
cm_results = [r for r in all_results if r.get('cm') is not None]
if len(cm_results) == 1: axes = [axes]
for ax, r in zip(axes, cm_results):
    cm = r['cm'].astype(float)
    cm_n = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=state_names_list, yticklabels=state_names_list, ax=ax, vmin=0, vmax=1, cbar=False)
    ax.set_title(f"{r['model']} (F1={r['f1']:.3f})", fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("✅ ALL DONE!")
print("="*60)
print(f"\nFiles for Mehul (in {RESULTS_DIR}/):")
print(f"  1. labels/hmm_patch_labels_val.npy  ({len(hmm_patch['val'])} patches)")
print(f"  2. labels/hmm_patch_labels_test.npy ({len(hmm_patch['test'])} patches)")
print(f"  3. results_table.csv")
print(f"\nMehul needs to:")
print(f"  1. Load these labels")
print(f"  2. Get FinJEPA representations (384-dim, same patch structure)")
print(f"  3. Run same linear probe evaluation")
