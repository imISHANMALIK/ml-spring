"""
FinJEPA — Master Pipeline (Dense Sliding Windows)
==================================================
This script runs the entire pipeline end-to-end using dense data augmentation.
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path
import pandas as pd

src_dir = Path(__file__).parent / "src" if "__file__" in dir() else Path("src")
sys.path.insert(0, str(src_dir))

CONFIG = {
    'patch_size': 20,
    'context_patches': 12,
    'hmm_n_states': 3,
    'supervised_epochs': 100,
    'supervised_lr': 1e-4,
    'supervised_batch_size': 64,
    'ts2vec_epochs': 200,
    'ts2vec_output_dims': 384,
    'patchtst_epochs': 200,
    'patchtst_d_model': 384,
    'finjepa_epochs': 200,
    'finjepa_d_model': 384,
    'device': 'auto',
    'results_dir': 'results',
}

def get_device():
    if torch.cuda.is_available(): return 'cuda'
    elif torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    results_dir = Path(CONFIG['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ═══════════════════════════════════════════
    # PHASE 1: Data Pipeline
    # ═══════════════════════════════════════════
    from data_pipeline import load_and_preprocess, create_dataloaders, DenseRegimeDataset
    data = load_and_preprocess()
    
    # ═══════════════════════════════════════════
    # PHASE 2: HMM Ground Truth Labels
    # ═══════════════════════════════════════════
    from hmm_labels import RegimeLabeler, export_labels
    labeler = RegimeLabeler(n_states=CONFIG['hmm_n_states'])
    # Fit HMM on daily z_returns
    train_returns = data['daily_returns']['train']['z_return'].values
    labeler.fit(train_returns)
    
    daily_labels = {}
    for split in ['train', 'val', 'test']:
        daily_labels[split] = labeler.predict(data['daily_returns'][split]['z_return'].values)
    
    # ═══════════════════════════════════════════
    # PHASE 3: Supervised Baseline
    # ═══════════════════════════════════════════
    from supervised_baseline import SupervisedBaseline, train_supervised, extract_representations
    
    train_sup_ds = DenseRegimeDataset(data['daily_returns']['train']['z_return'].values, daily_labels['train'])
    val_sup_ds = DenseRegimeDataset(data['daily_returns']['val']['z_return'].values, daily_labels['val'])
    test_sup_ds = DenseRegimeDataset(data['daily_returns']['test']['z_return'].values, daily_labels['test'])
    
    sup_model = SupervisedBaseline(patch_size=CONFIG['patch_size'])
    sup_model, _ = train_supervised(
        sup_model, train_sup_ds, val_sup_ds,
        n_epochs=CONFIG['supervised_epochs'],
        batch_size=CONFIG['supervised_batch_size'],
        device=device
    )
    sup_val_reprs, _ = extract_representations(sup_model, val_sup_ds, device=device)
    sup_test_reprs, test_labels_ext = extract_representations(sup_model, test_sup_ds, device=device)
    
    # ═══════════════════════════════════════════
    # Dataloaders for SSL Models
    # ═══════════════════════════════════════════
    ssl_loaders = create_dataloaders(data['daily_returns'], batch_size=64, mode='full')
    jepa_loaders = create_dataloaders(data['daily_returns'], batch_size=64, mode='jepa')
    
    # ═══════════════════════════════════════════
    # PHASE 4: TS2Vec
    # ═══════════════════════════════════════════
    from ts2vec_baseline import train_ts2vec, extract_ts2vec_representations
    ts2vec_model = train_ts2vec(ssl_loaders['train'], n_epochs=CONFIG['ts2vec_epochs'], device=device)
    ts2vec_val_reprs = extract_ts2vec_representations(ts2vec_model, ssl_loaders['val'], device=device)
    ts2vec_test_reprs = extract_ts2vec_representations(ts2vec_model, ssl_loaders['test'], device=device)
    
    # ═══════════════════════════════════════════
    # PHASE 5: PatchTST
    # ═══════════════════════════════════════════
    from patchtst_baseline import train_patchtst, extract_patchtst_representations
    patchtst_model = train_patchtst(ssl_loaders['train'], n_epochs=CONFIG['patchtst_epochs'], device=device)
    patchtst_val_reprs = extract_patchtst_representations(patchtst_model, ssl_loaders['val'], device=device)
    patchtst_test_reprs = extract_patchtst_representations(patchtst_model, ssl_loaders['test'], device=device)
    
    # ═══════════════════════════════════════════
    # PHASE 6: FinJEPA
    # ═══════════════════════════════════════════
    from finjepa import train_finjepa, extract_finjepa_representations
    finjepa_model = train_finjepa(jepa_loaders['train'], n_epochs=CONFIG['finjepa_epochs'], device=device)
    finjepa_val_reprs = extract_finjepa_representations(finjepa_model, jepa_loaders['val'], device=device)
    finjepa_test_reprs = extract_finjepa_representations(finjepa_model, jepa_loaders['test'], device=device)
    
    # ═══════════════════════════════════════════
    # EVALUATION
    # ═══════════════════════════════════════════
    from evaluate import evaluate_all_models
    
    # Re-extract labels from val_sup_ds
    _, val_labels_ext = extract_representations(sup_model, val_sup_ds, device=device)
    
    # Hack the labels dict to pass to evaluate_all_models
    labels_dict = {'patch_labels': {'val': val_labels_ext, 'test': test_labels_ext}}
    
    representations = {
        'Supervised': {'val': sup_val_reprs,      'test': sup_test_reprs},
        'TS2Vec':     {'val': ts2vec_val_reprs,   'test': ts2vec_test_reprs},
        'PatchTST':   {'val': patchtst_val_reprs, 'test': patchtst_test_reprs},
        'FinJEPA':    {'val': finjepa_val_reprs,  'test': finjepa_test_reprs},
    }
    
    all_results, results_df = evaluate_all_models(representations, labels_dict, save_dir=results_dir)
    # ═══════════════════════════════════════════
    # PHASE 8: Layer-Wise Probing
    # ═══════════════════════════════════════════
    from evaluate import evaluate_layerwise_comparison
    layerwise_results = evaluate_layerwise_comparison(
        finjepa_model=finjepa_model,
        patchtst_model=patchtst_model,
        val_loader=ssl_loaders['val'],
        test_loader=ssl_loaders['test'],
        val_labels=val_labels_ext,
        test_labels=test_labels_ext
    )

    import pandas as pd
    for model_name, layer_results in layerwise_results.items():
        pd.DataFrame(layer_results).to_csv(results_dir / f"layerwise_{model_name.lower()}.csv", index=False)

    # ═══════════════════════════════════════════
    # PHASE 9: Emergence Plot
    # ═══════════════════════════════════════════
    from plot_emergence import plot_emergence
    fig = plot_emergence(
        finjepa_results=layerwise_results['FinJEPA'],
        patchtst_results=layerwise_results['PatchTST'],
        save_path=results_dir / "emergence_plot.pdf",
        show=False,
    )
    fig.savefig(results_dir / "emergence_plot.png", dpi=200, bbox_inches='tight', facecolor="#FFFBF5")

    np.savez(
        results_dir / "all_representations.npz",
        sup_val=sup_val_reprs, sup_test=sup_test_reprs,
        ts2vec_val=ts2vec_val_reprs, ts2vec_test=ts2vec_test_reprs,
        patchtst_val=patchtst_val_reprs, patchtst_test=patchtst_test_reprs,
        finjepa_val=finjepa_val_reprs, finjepa_test=finjepa_test_reprs,
    )
    results_df.to_csv(results_dir / "results_table.csv", index=False)
    
    print("\n✅ Done!")
    return all_results, results_df, layerwise_results

if __name__ == "__main__":
    main()
