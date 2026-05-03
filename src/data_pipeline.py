"""
Data Pipeline for FinJEPA Project
==================================
Downloads S&P 500 data, computes log returns, rolling z-score normalization,
and creates *Dense Sliding Windows* (stride=1 day) for massive data augmentation.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

TICKER = "^GSPC"
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"

TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"

PATCH_SIZE = 20           
CONTEXT_PATCHES = 12      
TARGET_PATCHES = 4        
ROLLING_WINDOW = 252      

def download_sp500(ticker=TICKER, start=START_DATE, end=END_DATE, cache_dir=None):
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "results" / "raw_data"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{ticker}_raw.csv"
    
    if cache_path.exists():
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(cache_path)
    return df

def compute_log_returns(df):
    close = df['Close'].values.astype(np.float64)
    close = np.where(close <= 0, np.nan, close)
    log_returns = np.log(close[1:] / close[:-1])
    dates = df.index[1:]
    return pd.DataFrame({'close': close[1:], 'log_return': log_returns}, index=dates)

def rolling_zscore(returns, window=ROLLING_WINDOW):
    series = pd.Series(returns)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    rolling_std = rolling_std.replace(0, np.nan)
    return ((series - rolling_mean) / rolling_std).values

def temporal_split(daily_df):
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)
    val_end = pd.Timestamp(VAL_END)
    test_start = pd.Timestamp(TEST_START)
    
    splits = {}
    splits['train'] = daily_df[daily_df.index <= train_end]
    splits['val'] = daily_df[(daily_df.index >= val_start) & (daily_df.index <= val_end)]
    splits['test'] = daily_df[daily_df.index >= test_start]
    return splits

class DensePatchSequenceDataset(Dataset):
    def __init__(self, daily_z_returns, patch_size=PATCH_SIZE, 
                 context_len=CONTEXT_PATCHES, target_len=TARGET_PATCHES, mode='jepa'):
        self.returns = torch.FloatTensor(daily_z_returns)
        self.patch_size = patch_size
        self.context_len = context_len
        self.target_len = target_len
        self.total_patches = context_len + target_len
        self.window_size = self.total_patches * patch_size
        self.mode = mode
        self.n_samples = max(0, len(self.returns) - self.window_size + 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        window = self.returns[idx : idx + self.window_size]
        patches = window.reshape(self.total_patches, self.patch_size)
        
        if self.mode == 'jepa':
            return patches[:self.context_len], patches[self.context_len:]
        else:
            return patches

class DenseRegimeDataset(Dataset):
    def __init__(self, daily_z_returns, daily_labels, patch_size=PATCH_SIZE, 
                 context_len=CONTEXT_PATCHES):
        self.returns = torch.FloatTensor(daily_z_returns)
        self.labels = torch.LongTensor(daily_labels)
        self.patch_size = patch_size
        self.context_len = context_len
        self.window_size = context_len * patch_size
        self.n_samples = max(0, len(self.returns) - self.window_size + 1)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        window = self.returns[idx : idx + self.window_size]
        context = window.reshape(self.context_len, self.patch_size)
        label = self.labels[idx + self.window_size - 1]
        return context, label

def load_and_preprocess(cache_dir=None):
    raw_df = download_sp500(cache_dir=cache_dir)
    returns_df = compute_log_returns(raw_df)
    returns_df['z_return'] = rolling_zscore(returns_df['log_return'].values)
    valid_returns = returns_df.dropna(subset=['z_return']).copy()
    daily_splits = temporal_split(valid_returns)
    
    return {
        'raw_df': raw_df,
        'returns_df': valid_returns,
        'daily_returns': daily_splits,
    }
