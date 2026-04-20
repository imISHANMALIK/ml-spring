"""
HMM Ground Truth Regime Labels (Track A)
==========================================
Fits a 3-state Gaussian HMM on training-period daily log returns,
then infers regime labels for all periods. Labels are sorted by
mean return: 0=Bear, 1=Sideways, 2=Bull.

These labels serve as ground truth for ALL model evaluations.

Usage:
    from src.data_pipeline import load_and_preprocess
    from src.hmm_labels import RegimeLabeler
    
    data = load_and_preprocess()
    labeler = RegimeLabeler(n_states=3)
    labels = labeler.fit_and_label(data)
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')


class RegimeLabeler:
    """3-State Gaussian HMM for market regime classification.
    
    States (sorted by mean return):
        0 = Bear   (negative mean, high volatility)
        1 = Sideways (near-zero mean, low volatility) 
        2 = Bull   (positive mean, moderate volatility)
    
    IMPORTANT: Fitted ONCE on training data only. Never refit on val/test.
    """
    
    def __init__(self, n_states=3, n_iter=200, random_state=42):
        self.n_states = n_states
        self.hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
            tol=1e-4
        )
        self.state_order = None  # Mapping from HMM states to sorted states
        self.is_fitted = False
    
    def fit(self, train_returns):
        """Fit HMM on training period daily log returns.
        
        Args:
            train_returns: np.array of shape (n_days,) — daily log returns
        """
        X = train_returns.reshape(-1, 1)
        self.hmm.fit(X)
        
        # Sort states by mean return: bear < sideways < bull
        means = self.hmm.means_.flatten()
        self.state_order = np.argsort(means)  # indices that sort by mean
        
        self.is_fitted = True
        
        # Print state statistics
        print("\nHMM States (sorted by mean return):")
        print("-" * 60)
        state_names = ['Bear', 'Sideways', 'Bull'] if self.n_states == 3 else \
                      ['Bear', 'Bull'] if self.n_states == 2 else \
                      [f'State {i}' for i in range(self.n_states)]
        
        for new_idx, old_idx in enumerate(self.state_order):
            mean = self.hmm.means_[old_idx, 0]
            var = self.hmm.covars_[old_idx, 0, 0]
            std = np.sqrt(var)
            print(f"  {state_names[new_idx]:>10}: mean={mean:+.6f}, "
                  f"std={std:.6f}, "
                  f"annualized_mean={mean*252:.2%}, "
                  f"annualized_vol={std*np.sqrt(252):.2%}")
        
        print(f"\nTransition matrix (sorted):")
        sorted_trans = self.hmm.transmat_[self.state_order][:, self.state_order]
        for i, name in enumerate(state_names):
            row = sorted_trans[i]
            print(f"  {name:>10}: {np.array2string(row, precision=3)}")
    
    def predict(self, returns):
        """Predict regime labels using frozen HMM.
        
        Args:
            returns: np.array of shape (n_days,) — daily log returns
            
        Returns:
            labels: np.array of shape (n_days,) — regime labels (sorted: 0=bear, 1=sideways, 2=bull)
        """
        assert self.is_fitted, "Must call fit() first!"
        
        X = returns.reshape(-1, 1)
        raw_labels = self.hmm.predict(X)
        
        # Remap to sorted order
        reverse_map = np.zeros(self.n_states, dtype=int)
        for new_idx, old_idx in enumerate(self.state_order):
            reverse_map[old_idx] = new_idx
        
        sorted_labels = reverse_map[raw_labels]
        return sorted_labels
    
    def predict_proba(self, returns):
        """Get regime probabilities using frozen HMM.
        
        Returns:
            probs: np.array of shape (n_days, n_states) — sorted state probabilities
        """
        assert self.is_fitted, "Must call fit() first!"
        
        X = returns.reshape(-1, 1)
        raw_probs = self.hmm.predict_proba(X)
        
        # Reorder columns to sorted state order
        sorted_probs = raw_probs[:, self.state_order]
        return sorted_probs
    
    def labels_to_patches(self, daily_labels, patch_size=20):
        """Convert daily labels to patch-level labels via majority vote.
        
        Args:
            daily_labels: np.array of shape (n_days,)
            patch_size: int, days per patch
            
        Returns:
            patch_labels: np.array of shape (n_patches,)
        """
        n_days = len(daily_labels)
        n_patches = n_days // patch_size
        truncated = n_patches * patch_size
        
        reshaped = daily_labels[:truncated].reshape(n_patches, patch_size)
        patch_labels = stats.mode(reshaped, axis=1, keepdims=False).mode
        
        return patch_labels.astype(int)
    
    def save(self, path):
        """Save fitted HMM to disk."""
        joblib.dump({
            'hmm': self.hmm,
            'state_order': self.state_order,
            'n_states': self.n_states,
            'is_fitted': self.is_fitted
        }, path)
        print(f"Saved HMM to {path}")
    
    @classmethod
    def load(cls, path):
        """Load fitted HMM from disk."""
        data = joblib.load(path)
        labeler = cls(n_states=data['n_states'])
        labeler.hmm = data['hmm']
        labeler.state_order = data['state_order']
        labeler.is_fitted = data['is_fitted']
        print(f"Loaded HMM from {path}")
        return labeler
    
    def fit_and_label(self, pipeline_data, patch_size=20):
        """Convenience: fit on train, label everything.
        
        Args:
            pipeline_data: dict from data_pipeline.load_and_preprocess()
            
        Returns:
            dict with keys:
                'daily_labels': {train, val, test} → np.array of daily labels
                'patch_labels': {train, val, test} → np.array of patch labels
                'daily_probs': {train, val, test} → np.array of probabilities
        """
        # Fit on training returns only
        train_returns = pipeline_data['daily_returns']['train']['log_return'].values
        self.fit(train_returns)
        
        result = {'daily_labels': {}, 'patch_labels': {}, 'daily_probs': {}}
        
        for split in ['train', 'val', 'test']:
            returns = pipeline_data['daily_returns'][split]['log_return'].values
            
            # Daily labels
            daily_labels = self.predict(returns)
            result['daily_labels'][split] = daily_labels
            
            # Daily probabilities
            daily_probs = self.predict_proba(returns)
            result['daily_probs'][split] = daily_probs
            
            # Patch-level labels (majority vote)
            patch_labels = self.labels_to_patches(daily_labels, patch_size)
            result['patch_labels'][split] = patch_labels
            
            # Print distribution
            unique, counts = np.unique(daily_labels, return_counts=True)
            state_names = ['Bear', 'Sideways', 'Bull'][:self.n_states]
            print(f"\n{split.upper()} regime distribution (daily):")
            for state, count in zip(unique, counts):
                pct = count / len(daily_labels) * 100
                print(f"  {state_names[state]:>10}: {count:5d} days ({pct:.1f}%)")
            
            unique_p, counts_p = np.unique(patch_labels, return_counts=True)
            print(f"{split.upper()} regime distribution (patches):")
            for state, count in zip(unique_p, counts_p):
                pct = count / len(patch_labels) * 100
                print(f"  {state_names[state]:>10}: {count:4d} patches ({pct:.1f}%)")
        
        return result


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_regimes(pipeline_data, labels_data, save_path=None):
    """Plot S&P 500 price with HMM regime labels overlaid.
    
    This is the key sanity check — do regimes capture:
    - 2008 financial crisis (bear)
    - 2020 COVID crash (bear)
    - 2022 bear market (bear)
    - Bull runs in between
    """
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), 
                              gridspec_kw={'height_ratios': [3, 1, 1]})
    
    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}  # bear=red, sideways=yellow, bull=green
    state_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
    
    # Combine all daily data
    all_splits = []
    for split in ['train', 'val', 'test']:
        df = pipeline_data['daily_returns'][split].copy()
        df['regime'] = labels_data['daily_labels'][split]
        df['split'] = split
        all_splits.append(df)
    
    combined = pd.concat(all_splits)
    
    # Panel 1: Price with regime coloring
    ax1 = axes[0]
    ax1.set_title('S&P 500 with HMM Regime Labels', fontsize=14, fontweight='bold')
    
    for regime in [0, 1, 2]:
        mask = combined['regime'] == regime
        dates = combined.index[mask]
        prices = combined['close'][mask]
        ax1.scatter(dates, prices, c=colors[regime], s=1, alpha=0.7, 
                   label=state_names[regime])
    
    # Mark split boundaries
    for date, label in [(TRAIN_END, 'Train→Val'), (VAL_END, 'Val→Test')]:
        ax1.axvline(pd.Timestamp(date), color='white', linestyle='--', alpha=0.5)
        ax1.text(pd.Timestamp(date), ax1.get_ylim()[1] * 0.95, label, 
                fontsize=9, ha='center', color='white')
    
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left', markerscale=10)
    ax1.set_facecolor('#1a1a2e')
    ax1.grid(True, alpha=0.2)
    
    # Panel 2: Regime timeline
    ax2 = axes[1]
    ax2.set_title('Regime Timeline', fontsize=12)
    
    for regime in [0, 1, 2]:
        mask = combined['regime'] == regime
        dates = combined.index[mask]
        ax2.scatter(dates, [regime] * len(dates), c=colors[regime], s=2, alpha=0.8)
    
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Bear', 'Sideways', 'Bull'])
    ax2.set_facecolor('#1a1a2e')
    ax2.grid(True, alpha=0.2)
    
    # Panel 3: Daily returns colored by regime
    ax3 = axes[2]
    ax3.set_title('Daily Returns by Regime', fontsize=12)
    
    for regime in [0, 1, 2]:
        mask = combined['regime'] == regime
        dates = combined.index[mask]
        returns = combined['log_return'][mask]
        ax3.scatter(dates, returns, c=colors[regime], s=1, alpha=0.5)
    
    ax3.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    ax3.set_ylabel('Log Return')
    ax3.set_facecolor('#1a1a2e')
    ax3.grid(True, alpha=0.2)
    
    fig.patch.set_facecolor('#0f0f1e')
    for ax in axes:
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved regime plot to {save_path}")
    
    plt.show()
    return fig


def export_labels(labels_data, output_dir):
    """Export labels as .npy files for Mehul (FinJEPA).
    
    Mehul needs:
        - val patch labels (to train linear probe)
        - test patch labels (to evaluate)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        # Daily labels
        path = output_dir / f"hmm_daily_labels_{split}.npy"
        np.save(path, labels_data['daily_labels'][split])
        
        # Patch labels
        path = output_dir / f"hmm_patch_labels_{split}.npy"
        np.save(path, labels_data['patch_labels'][split])
        
        # Daily probabilities
        path = output_dir / f"hmm_daily_probs_{split}.npy"
        np.save(path, labels_data['daily_probs'][split])
    
    print(f"\nExported all labels to {output_dir}/")
    print("Files for Mehul:")
    print(f"  - hmm_patch_labels_val.npy   ({len(labels_data['patch_labels']['val'])} patches)")
    print(f"  - hmm_patch_labels_test.npy  ({len(labels_data['patch_labels']['test'])} patches)")


# ─────────────────────────────────────────────
# Constants for import
# ─────────────────────────────────────────────
TRAIN_END = "2019-12-31"
VAL_END = "2021-12-31"


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from data_pipeline import load_and_preprocess
    
    # Load data
    data = load_and_preprocess()
    
    # Fit HMM and label everything
    labeler = RegimeLabeler(n_states=3)
    labels = labeler.fit_and_label(data)
    
    # Save model
    results_dir = Path(__file__).parent.parent / "results"
    labeler.save(results_dir / "hmm_model.pkl")
    
    # Export labels for Mehul
    export_labels(labels, results_dir / "labels")
    
    # Visualize
    plot_regimes(data, labels, save_path=results_dir / "regime_plot.png")
