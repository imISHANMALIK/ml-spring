"""
Data Pipeline for FinJEPA Project
==================================
Downloads Dow 30 data, computes log returns, rolling z-score normalization,
creates 20-day patches, and performs strict temporal splitting.

Usage:
    from src.data_pipeline import load_and_preprocess, create_patch_datasets
    
    data = load_and_preprocess()
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DOW_30 = [
    "AAPL", "MSFT", "JPM", "V", "PG", "UNH", "JNJ", "HD", "CVX", "MRK", 
    "KO", "CSCO", "MCD", "WMT", "CRM", "INTC", "VZ", "IBM", "WBA", "BA", 
    "HON", "AMGN", "CAT", "NKE", "AXP", "DIS", "MMM", "TRV", "GS", "DOW"
]
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

# Temporal split boundaries (strict, no leakage)
TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"

# Patch configuration
PATCH_SIZE = 20           # 20 trading days ≈ 1 month
CONTEXT_PATCHES = 12      # 12 patches = 240 trading days lookback
TARGET_PATCHES = 4        # 4 patches = 80 trading days forward
ROLLING_WINDOW = 252      # 1 trading year for z-score normalization


# ─────────────────────────────────────────────
# Data download and feature engineering
# ─────────────────────────────────────────────
def download_ticker(ticker, start=START_DATE, end=END_DATE, cache_dir=None):
    """Download OHLCV data via yfinance.
    
    Caches to disk to avoid re-downloading.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "results" / "raw_data"
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_raw.csv"
    
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
    
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if len(df) == 0:
        print(f"Warning: No data for {ticker}")
        return df
        
    df.to_csv(cache_path)
    
    return df


def compute_log_returns(df):
    """Compute log returns from close prices.
    
    log_return_t = log(close_t / close_{t-1})
    """
    close = df['Close'].values.astype(np.float64)
    # Handle zeros or negatives to avoid log issues
    close = np.where(close <= 0, np.nan, close)
    log_returns = np.log(close[1:] / close[:-1])
    
    # Create DataFrame aligned with dates (drop first row since we lose one)
    dates = df.index[1:]
    result = pd.DataFrame({
        'close': close[1:],
        'log_return': log_returns
    }, index=dates)
    
    return result


def rolling_zscore(returns, window=ROLLING_WINDOW):
    """Rolling z-score normalization (strictly causal — lookback only).
    
    z_t = (r_t - mean(r_{t-window:t})) / std(r_{t-window:t})
    
    This is critical: no future data leaks into normalization.
    """
    series = pd.Series(returns)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_scores = (series - rolling_mean) / rolling_std
    
    return z_scores.values


def create_patches(data, patch_size=PATCH_SIZE):
    """Create non-overlapping patches from normalized returns.
    
    Each patch is a vector of (patch_size,) normalized returns.
    Returns:
        patches: np.array of shape (n_patches, patch_size)
        patch_dates: list of (start_date, end_date) per patch
    """
    # Drop NaNs from rolling z-score warmup
    valid_mask = ~np.isnan(data['z_return'].values)
    valid_data = data[valid_mask].copy()
    
    if len(valid_data) < patch_size:
        return np.array([]), np.array([]), []
        
    n_days = len(valid_data)
    n_patches = n_days // patch_size
    
    # Truncate to exact multiple of patch_size
    truncated = n_days - (n_days % patch_size)
    
    returns = valid_data['z_return'].values[:truncated]
    raw_returns = valid_data['log_return'].values[:truncated]
    dates = valid_data.index[:truncated]
    
    patches = returns.reshape(n_patches, patch_size)
    raw_patches = raw_returns.reshape(n_patches, patch_size)
    
    # Track date ranges for each patch
    patch_dates = []
    for i in range(n_patches):
        start_idx = i * patch_size
        end_idx = (i + 1) * patch_size - 1
        patch_dates.append((dates[start_idx], dates[end_idx]))
    
    return patches, raw_patches, patch_dates


# ─────────────────────────────────────────────
# Temporal splitting
# ─────────────────────────────────────────────
def temporal_split(patches, patch_dates, 
                   train_end=TRAIN_END, val_start=VAL_START, 
                   val_end=VAL_END, test_start=TEST_START):
    """Split patches by date into train/val/test.
    
    Uses the END date of each patch to determine which split it belongs to.
    This ensures no future leakage.
    """
    train_end = pd.Timestamp(train_end)
    val_start = pd.Timestamp(val_start)
    val_end = pd.Timestamp(val_end)
    test_start = pd.Timestamp(test_start)
    
    train_idx, val_idx, test_idx = [], [], []
    
    for i, (start, end) in enumerate(patch_dates):
        if end <= train_end:
            train_idx.append(i)
        elif start >= val_start and end <= val_end:
            val_idx.append(i)
        elif start >= test_start:
            test_idx.append(i)
        # Patches straddling boundaries are dropped
    
    splits = {
        'train': {
            'patches': patches[train_idx] if len(train_idx) > 0 else np.array([]),
            'indices': train_idx,
            'dates': [patch_dates[i] for i in train_idx]
        },
        'val': {
            'patches': patches[val_idx] if len(val_idx) > 0 else np.array([]),
            'indices': val_idx,
            'dates': [patch_dates[i] for i in val_idx]
        },
        'test': {
            'patches': patches[test_idx] if len(test_idx) > 0 else np.array([]),
            'indices': test_idx,
            'dates': [patch_dates[i] for i in test_idx]
        }
    }
    
    return splits


# ─────────────────────────────────────────────
# PyTorch Datasets
# ─────────────────────────────────────────────
class PatchSequenceDataset(Dataset):
    """Dataset that returns sequences of patches for JEPA-style training.
    
    Each sample is:
        context_patches: (context_len, patch_size) — past patches
        target_patches: (target_len, patch_size) — future patches to predict
    
    For baselines (TS2Vec, PatchTST), the full sequence is used.
    """
    
    def __init__(self, patches, context_len=CONTEXT_PATCHES, 
                 target_len=TARGET_PATCHES, mode='jepa'):
        """
        Args:
            patches: np.array of shape (n_patches, patch_size)
            context_len: number of context patches (default 12 = 240 days)
            target_len: number of target patches (default 4 = 80 days)
            mode: 'jepa' returns (context, target), 
                  'full' returns full sequence for baselines
        """
        self.patches = torch.FloatTensor(patches)
        self.context_len = context_len
        self.target_len = target_len
        self.total_len = context_len + target_len
        self.mode = mode
        
        # Number of valid sliding windows
        self.n_samples = max(0, len(patches) - self.total_len + 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.mode == 'jepa':
            context = self.patches[idx:idx + self.context_len]
            target = self.patches[idx + self.context_len:idx + self.total_len]
            return context, target
        else:
            # Full sequence for baselines
            full_seq = self.patches[idx:idx + self.total_len]
            return full_seq


class RegimeDataset(Dataset):
    """Dataset for supervised baseline — patches with regime labels.
    
    Each sample is:
        patches: (context_len, patch_size) — input patches
        label: int — regime label (0=bear, 1=sideways, 2=bull)
    """
    
    def __init__(self, patches, labels, context_len=CONTEXT_PATCHES):
        """
        Args:
            patches: np.array of shape (n_patches, patch_size)
            labels: np.array of shape (n_patches,) — per-patch regime labels
            context_len: number of patches per input sequence
        """
        self.patches = torch.FloatTensor(patches)
        self.labels = torch.LongTensor(labels)
        self.context_len = context_len
        
        # We predict the regime of the LAST patch in the context window
        self.n_samples = max(0, len(patches) - context_len + 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        context = self.patches[idx:idx + self.context_len]
        # Label is the regime of the last patch in the window
        label = self.labels[idx + self.context_len - 1]
        return context, label


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def load_and_preprocess(cache_dir=None):
    """Full pipeline: download → log returns → z-score → patches → split.
    
    Returns:
        dict with keys:
            'patches': np.array (n_patches, patch_size)
            'patch_dates': list of (start, end) date tuples
            'splits': dict with train/val/test patches
            'daily_returns': dict with train/val/test daily log returns
    """
    all_train_patches = []
    all_val_patches = []
    all_test_patches = []
    
    all_train_dates = []
    all_val_dates = []
    all_test_dates = []
    
    daily_returns_train = []
    daily_returns_val = []
    daily_returns_test = []
    
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    val_end = pd.Timestamp(VAL_END)
    test_start = pd.Timestamp(TEST_START)
    
    print(f"Processing {len(DOW_30)} tickers...")
    
    for ticker in DOW_30:
        # 1. Download
        df = download_ticker(ticker, cache_dir=cache_dir)
        if len(df) == 0:
            continue
            
        # 2. Log returns
        returns_df = compute_log_returns(df)
        
        # 3. Rolling z-score
        returns_df['z_return'] = rolling_zscore(returns_df['log_return'].values)
        
        # 4. Create patches
        patches, raw_patches, patch_dates = create_patches(returns_df)
        if len(patches) == 0:
            continue
            
        # 5. Temporal split (patches)
        splits = temporal_split(patches, patch_dates)
        
        if len(splits['train']['patches']) > 0:
            all_train_patches.append(splits['train']['patches'])
            all_train_dates.extend(splits['train']['dates'])
            
        if len(splits['val']['patches']) > 0:
            all_val_patches.append(splits['val']['patches'])
            all_val_dates.extend(splits['val']['dates'])
            
        if len(splits['test']['patches']) > 0:
            all_test_patches.append(splits['test']['patches'])
            all_test_dates.extend(splits['test']['dates'])
        
        # 6. Also split DAILY returns for HMM fitting
        valid_returns = returns_df.dropna(subset=['z_return'])
        
        daily_train = valid_returns[valid_returns.index <= train_end]
        daily_val = valid_returns[(valid_returns.index >= val_start) & (valid_returns.index <= val_end)]
        daily_test = valid_returns[valid_returns.index >= test_start]
        
        if len(daily_train) > 0:
            daily_returns_train.append(daily_train)
        if len(daily_val) > 0:
            daily_returns_val.append(daily_val)
        if len(daily_test) > 0:
            daily_returns_test.append(daily_test)
            
    # Combine everything
    final_train_patches = np.concatenate(all_train_patches, axis=0) if all_train_patches else np.array([])
    final_val_patches = np.concatenate(all_val_patches, axis=0) if all_val_patches else np.array([])
    final_test_patches = np.concatenate(all_test_patches, axis=0) if all_test_patches else np.array([])
    
    master_splits = {
        'train': {'patches': final_train_patches, 'dates': all_train_dates},
        'val': {'patches': final_val_patches, 'dates': all_val_dates},
        'test': {'patches': final_test_patches, 'dates': all_test_dates}
    }
    
    # Combine all patches for full array if needed (rarely used now)
    all_patches = np.concatenate([final_train_patches, final_val_patches, final_test_patches], axis=0)
    all_dates = all_train_dates + all_val_dates + all_test_dates
    
    # Combine daily returns
    master_daily = {
        'train': pd.concat(daily_returns_train) if daily_returns_train else pd.DataFrame(),
        'val': pd.concat(daily_returns_val) if daily_returns_val else pd.DataFrame(),
        'test': pd.concat(daily_returns_test) if daily_returns_test else pd.DataFrame()
    }
    
    print(f"\nCompleted Processing!")
    print(f"Patches -> Train: {len(final_train_patches)}, Val: {len(final_val_patches)}, Test: {len(final_test_patches)}")
    print(f"Daily returns -> Train: {len(master_daily['train'])}, Val: {len(master_daily['val'])}, Test: {len(master_daily['test'])}")
    
    return {
        'patches': all_patches,
        'patch_dates': all_dates,
        'splits': master_splits,
        'daily_returns': master_daily,
    }


def create_dataloaders(splits, batch_size=32, mode='jepa'):
    """Create PyTorch DataLoaders from split patches."""
    loaders = {}
    for split_name, split_data in splits.items():
        dataset = PatchSequenceDataset(
            split_data['patches'], mode=mode
        )
        loaders[split_name] = DataLoader(
            dataset, batch_size=batch_size, 
            shuffle=(split_name == 'train'),
            drop_last=(split_name == 'train')
        )
    return loaders


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data = load_and_preprocess()
    
    print(f"\nTotal patches: {len(data['patches'])}")
    print(f"Patch shape: {data['patches'].shape}")
    
    for split_name, split_data in data['splits'].items():
        if len(split_data['patches']) > 0:
            print(f"  {split_name}: {len(split_data['patches'])} patches")
