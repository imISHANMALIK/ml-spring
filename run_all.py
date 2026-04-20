"""
FinJEPA — Master Pipeline (Run on Google Colab)
==================================================
This script runs the entire pipeline end-to-end:
    1. Data pipeline: download S&P 500, preprocess, patch, split
    2. Track A: Fit HMM, generate ground truth labels
    3. Supervised baseline: train and extract representations
    4. Track D: TS2Vec baseline — train and extract representations
    5. Track E: PatchTST baseline — train and extract representations
    6. Unified evaluation: linear probe + metrics + UMAP + results table

Run on Colab:
    1. Upload the entire `src/` folder to Colab
    2. !pip install yfinance hmmlearn torch transformers umap-learn scikit-learn matplotlib seaborn
    3. Run this script

Or run locally:
    cd ml-spring
    pip install -r requirements.txt
    python run_all.py
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / "src" if "__file__" in dir() else Path("src")
sys.path.insert(0, str(src_dir))

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CONFIG = {
    # Data
    'patch_size': 20,
    'context_patches': 12,
    
    # HMM
    'hmm_n_states': 3,
    
    # Supervised baseline
    'supervised_epochs': 100,
    'supervised_lr': 1e-4,
    'supervised_batch_size': 32,
    
    # TS2Vec
    'ts2vec_epochs': 200,
    'ts2vec_output_dims': 384,
    
    # PatchTST
    'patchtst_epochs': 200,
    'patchtst_d_model': 384,
    
    # Device
    'device': 'auto',
    
    # Output
    'results_dir': 'results',
}


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    """Run the full pipeline."""
    device = get_device()
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    results_dir = Path(CONFIG['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════
    # PHASE 1: Data Pipeline
    # ═══════════════════════════════════════════
    print("\n" + "█"*60)
    print("PHASE 1: DATA PIPELINE")
    print("█"*60)
    
    from data_pipeline import load_and_preprocess, RegimeDataset
    
    data = load_and_preprocess()
    
    train_patches = data['splits']['train']['patches']
    val_patches = data['splits']['val']['patches']
    test_patches = data['splits']['test']['patches']
    
    print(f"\nPatches — Train: {train_patches.shape}, "
          f"Val: {val_patches.shape}, Test: {test_patches.shape}")
    
    # ═══════════════════════════════════════════
    # PHASE 2: HMM Ground Truth Labels
    # ═══════════════════════════════════════════
    print("\n" + "█"*60)
    print("PHASE 2: HMM GROUND TRUTH LABELS (Track A)")
    print("█"*60)
    
    from hmm_labels import RegimeLabeler, plot_regimes, export_labels
    
    labeler = RegimeLabeler(n_states=CONFIG['hmm_n_states'])
    labels = labeler.fit_and_label(data, patch_size=CONFIG['patch_size'])
    
    # Save model and export labels for Mehul
    labeler.save(results_dir / "hmm_model.pkl")
    export_labels(labels, results_dir / "labels")
    
    # Visualize regimes
    try:
        plot_regimes(data, labels, save_path=results_dir / "regime_plot.png")
    except Exception as e:
        print(f"Plotting failed (non-critical): {e}")
    
    # Compute 5-day forward returns for forecasting metric
    def compute_forward_returns(patches, window=5):
        """Compute the cumulative return of each patch (sum of daily returns)."""
        return patches.sum(axis=1)
    
    forward_returns = {
        'train': compute_forward_returns(train_patches),
        'val': compute_forward_returns(val_patches),
        'test': compute_forward_returns(test_patches),
    }
    
    # ═══════════════════════════════════════════
    # PHASE 3: Supervised Baseline
    # ═══════════════════════════════════════════
    print("\n" + "█"*60)
    print("PHASE 3: SUPERVISED BASELINE")
    print("█"*60)
    
    from supervised_baseline import (SupervisedBaseline, train_supervised, 
                                     extract_representations)
    
    # Create supervised datasets
    train_ds = RegimeDataset(
        train_patches, labels['patch_labels']['train'],
        context_len=CONFIG['context_patches']
    )
    val_ds = RegimeDataset(
        val_patches, labels['patch_labels']['val'],
        context_len=CONFIG['context_patches']
    )
    test_ds = RegimeDataset(
        test_patches, labels['patch_labels']['test'],
        context_len=CONFIG['context_patches']
    )
    
    print(f"Supervised datasets — Train: {len(train_ds)}, "
          f"Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    sup_model = SupervisedBaseline(patch_size=CONFIG['patch_size'])
    sup_model, sup_history = train_supervised(
        sup_model, train_ds, val_ds,
        n_epochs=CONFIG['supervised_epochs'],
        lr=CONFIG['supervised_lr'],
        batch_size=CONFIG['supervised_batch_size'],
        device=device
    )
    
    # Extract representations
    sup_val_reprs, sup_val_labels = extract_representations(sup_model, val_ds, device=device)
    sup_test_reprs, sup_test_labels = extract_representations(sup_model, test_ds, device=device)
    
    # Save
    torch.save(sup_model.state_dict(), results_dir / "supervised_model.pt")
    
    # ═══════════════════════════════════════════
    # PHASE 4: TS2Vec Baseline (Track D)
    # ═══════════════════════════════════════════
    print("\n" + "█"*60)
    print("PHASE 4: TS2VEC BASELINE (Track D)")
    print("█"*60)
    
    from ts2vec_baseline import train_ts2vec, extract_ts2vec_representations
    
    ts2vec_model = train_ts2vec(
        train_patches, 
        output_dims=CONFIG['ts2vec_output_dims'],
        n_epochs=CONFIG['ts2vec_epochs'],
        device=device
    )
    
    ts2vec_val_reprs = extract_ts2vec_representations(ts2vec_model, val_patches)
    ts2vec_test_reprs = extract_ts2vec_representations(ts2vec_model, test_patches)
    
    # Save
    ts2vec_model.save(results_dir / "ts2vec_model.pt")
    
    # ═══════════════════════════════════════════
    # PHASE 5: PatchTST Baseline (Track E)
    # ═══════════════════════════════════════════
    print("\n" + "█"*60)
    print("PHASE 5: PATCHTST BASELINE (Track E)")
    print("█"*60)
    
    from patchtst_baseline import train_patchtst, extract_patchtst_representations
    
    patchtst_model = train_patchtst(
        train_patches,
        d_model=CONFIG['patchtst_d_model'],
        n_epochs=CONFIG['patchtst_epochs'],
        device=device
    )
    
    patchtst_val_reprs = extract_patchtst_representations(patchtst_model, val_patches)
    patchtst_test_reprs = extract_patchtst_representations(patchtst_model, test_patches)
    
    # Save
    patchtst_model.save(results_dir / "patchtst_model.pt")
    
    # ═══════════════════════════════════════════
    # PHASE 6: Unified Evaluation
    # ═══════════════════════════════════════════
    print("\n" + "█"*60)
    print("PHASE 6: UNIFIED EVALUATION")
    print("█"*60)
    
    from evaluate import evaluate_all_models
    
    # Package all representations
    representations = {
        'Supervised': {
            'val': sup_val_reprs,
            'test': sup_test_reprs,
        },
        'TS2Vec': {
            'val': ts2vec_val_reprs,
            'test': ts2vec_test_reprs,
        },
        'PatchTST': {
            'val': patchtst_val_reprs,
            'test': patchtst_test_reprs,
        },
    }
    
    # Align forward returns with representation counts
    # (TS2Vec and PatchTST produce fewer outputs due to context windowing)
    
    all_results, results_df = evaluate_all_models(
        representations, labels,
        forward_returns=forward_returns,
        save_dir=results_dir
    )
    
    # ═══════════════════════════════════════════
    # Save everything
    # ═══════════════════════════════════════════
    print("\n" + "█"*60)
    print("SAVING RESULTS")
    print("█"*60)
    
    # Save representations for Mehul to compare with FinJEPA
    np.savez(
        results_dir / "all_representations.npz",
        sup_val=sup_val_reprs,
        sup_test=sup_test_reprs,
        ts2vec_val=ts2vec_val_reprs,
        ts2vec_test=ts2vec_test_reprs,
        patchtst_val=patchtst_val_reprs,
        patchtst_test=patchtst_test_reprs,
    )
    
    results_df.to_csv(results_dir / "results_table.csv", index=False)
    
    print(f"\n✅ All results saved to {results_dir}/")
    print(f"\nFiles for Mehul:")
    print(f"  1. {results_dir}/labels/hmm_patch_labels_val.npy")
    print(f"  2. {results_dir}/labels/hmm_patch_labels_test.npy")
    print(f"  3. {results_dir}/hmm_model.pkl")
    print(f"  4. {results_dir}/all_representations.npz (for comparison)")
    print(f"\nMehul should:")
    print(f"  1. Load HMM labels from results/labels/")
    print(f"  2. Extract FinJEPA representations (384-dim)")
    print(f"  3. Run the same evaluate.py to compare")
    
    return all_results, results_df


if __name__ == "__main__":
    results, df = main()
