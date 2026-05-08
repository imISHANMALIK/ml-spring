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

# Must set path BEFORE importing project modules
src_dir = Path(__file__).parent / "src" if "__file__" in dir() else Path("src")
sys.path.insert(0, str(src_dir))

import data_pipeline
from data_pipeline import load_and_preprocess, create_dataloaders, DenseRegimeDataset
from hmm_labels import RegimeLabeler, export_labels
from supervised_baseline import SupervisedBaseline, train_supervised, extract_representations
from ts2vec_baseline import train_ts2vec, extract_ts2vec_representations
from patchtst_baseline import train_patchtst, extract_patchtst_representations
from finjepa import train_finjepa, extract_finjepa_representations, FinJEPAModel
from evaluate import (evaluate_all_models, evaluate_model, evaluate_random_baseline,
                      generate_results_table, plot_umap_comparison, plot_confusion_matrices,
                      evaluate_layerwise_comparison)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score
from plot_emergence import plot_emergence

CONFIG = {
    'patch_size': 20,
    'seed': 12,
    'context_patches': 12,
    'hmm_n_states': 3,
    'supervised_epochs': 100,
    'supervised_lr': 1e-4,
    'supervised_batch_size': 64,
    'ts2vec_epochs': 400,
    'ts2vec_output_dims': 384,
    'patchtst_epochs': 400,
    'patchtst_d_model': 384,
    'finjepa_epochs': 400,
    'finjepa_d_model': 384,
    'device': 'auto',
    'results_dir': 'results',
}

def get_device():
    if torch.cuda.is_available(): return 'cuda'
    elif torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

def main():
    # ── Set global seed for reproducibility ──────────────────────
    SEED = 12
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    import random
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {SEED}")
    # ─────────────────────────────────────────────────────────────

    device = get_device()
    print(f"Using device: {device}")

    # ── Helper functions ─────────────────────────────────────────
    def extract_sup_on_ssl_windows(model, loader, device):
        """Extract supervised representations on SSL-aligned windows.
        Takes first 12 patches (240 days) of each 320-day SSL window.
        Ensures all models evaluated on identical time windows."""
        model.eval()
        reprs = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x_ctx = x[:, :12, :].to(device)  # first 12 patches only
                r = model.get_representations(x_ctx)
                reprs.append(r.cpu().numpy())
        return np.concatenate(reprs, axis=0)

    results_dir = Path(CONFIG['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════
    # PHASE 1: Data Pipeline
    # ═══════════════════════════════════════════
    data = load_and_preprocess()
    print("\n" + "="*60)
    print("PHASE 1 SUMMARY")
    print("="*60)
    print(f"  STRIDE:       {data_pipeline.STRIDE}")
    print(f"  Train days:   {len(data['daily_returns']['train'])}")
    print(f"  Val days:     {len(data['daily_returns']['val'])}")
    print(f"  Test days:    {len(data['daily_returns']['test'])}")
    print(f"  Date ranges:")
    print(f"    Train: {data['daily_returns']['train'].index[0].date()} → {data['daily_returns']['train'].index[-1].date()}")
    print(f"    Val:   {data['daily_returns']['val'].index[0].date()} → {data['daily_returns']['val'].index[-1].date()}")
    print(f"    Test:  {data['daily_returns']['test'].index[0].date()} → {data['daily_returns']['test'].index[-1].date()}")


    # ═══════════════════════════════════════════
    # PHASE 2: HMM Ground Truth Labels
    # ═══════════════════════════════════════════
    labeler = RegimeLabeler(n_states=CONFIG['hmm_n_states'])
    
    train_returns = data['daily_returns']['train']['log_return'].values
    labeler.fit(train_returns)

    daily_labels = {}
    for split in ['train', 'val', 'test']:
        daily_labels[split] = labeler.predict(
            data['daily_returns'][split]['log_return'].values
        )
   
    def get_window_labels(daily_labels_arr, window_days, stride=5):
        """Get one label per sliding window — label of last day in window."""
        n = max(0, (len(daily_labels_arr) - window_days) // stride + 1)
        return np.array([
            daily_labels_arr[i * stride + window_days - 1]
            for i in range(n)
        ])

    PATCH_SIZE   = 20
    CONTEXT_LEN  = 12
    TARGET_LEN   = 4
    STRIDE       = 5

    # SSL models: full window = (context + target) patches = 320 days
    SSL_WINDOW   = (CONTEXT_LEN + TARGET_LEN) * PATCH_SIZE   # 320

    # Supervised: context only = 240 days
    SUP_WINDOW   = CONTEXT_LEN * PATCH_SIZE                   # 240

    # Generate correctly aligned labels for each model type
    val_labels_ssl   = get_window_labels(daily_labels['val'],  SSL_WINDOW, STRIDE)
    test_labels_ssl  = get_window_labels(daily_labels['test'], SSL_WINDOW, STRIDE)
    val_labels_sup   = get_window_labels(daily_labels['val'],  SUP_WINDOW, STRIDE)
    test_labels_sup  = get_window_labels(daily_labels['test'], SUP_WINDOW, STRIDE)

    # FinJEPA uses jepa_loader (mode='jepa') — same window size as SSL (320 days)
    # but label should correspond to end of CONTEXT window, not end of full window
    def get_jepa_labels(daily_labels_arr, stride=5):
        """Label = last day of context window (day 240), not day 320."""
        total_window = (CONTEXT_LEN + TARGET_LEN) * PATCH_SIZE  # 320
        context_end  = CONTEXT_LEN * PATCH_SIZE                  # 240
        n = max(0, (len(daily_labels_arr) - total_window) // stride + 1)
        return np.array([
            daily_labels_arr[i * stride + context_end - 1]
            for i in range(n)
        ])

    val_labels_jepa  = get_jepa_labels(daily_labels['val'])
    test_labels_jepa = get_jepa_labels(daily_labels['test'])

    print(f"Label counts — SSL: val={len(val_labels_ssl)}, test={len(test_labels_ssl)}")
    print(f"Label counts — Sup: val={len(val_labels_sup)}, test={len(test_labels_sup)}")
    print(f"Label counts — JEPA: val={len(val_labels_jepa)}, test={len(test_labels_jepa)}")

    print("\n" + "="*60)
    print("PHASE 2 SUMMARY")
    print("="*60)
    for split in ['train', 'val', 'test']:
        unique, counts = np.unique(daily_labels[split], return_counts=True)
        total = len(daily_labels[split])
        print(f"  {split.upper()} label distribution ({total} days):")
        for u, c in zip(unique, counts):
            name = ['Bear', 'Sideways', 'Bull'][u]
            print(f"    {name}: {c} ({c/total*100:.1f}%)")

    print(f"\n  Window label counts:")
    print(f"    SSL  — val: {len(val_labels_ssl)}, test: {len(test_labels_ssl)}")
    print(f"    Sup  — val: {len(val_labels_sup)}, test: {len(test_labels_sup)}")
    print(f"    JEPA — val: {len(val_labels_jepa)}, test: {len(test_labels_jepa)}")

    print(f"\n  SSL val label distribution:")
    unique, counts = np.unique(val_labels_ssl, return_counts=True)
    for u, c in zip(unique, counts):
        name = ['Bear', 'Sideways', 'Bull'][u]
        print(f"    {name}: {c} ({c/len(val_labels_ssl)*100:.1f}%)")

    print(f"\n  SSL test label distribution:")
    unique, counts = np.unique(test_labels_ssl, return_counts=True)
    for u, c in zip(unique, counts):
        name = ['Bear', 'Sideways', 'Bull'][u]
        print(f"    {name}: {c} ({c/len(test_labels_ssl)*100:.1f}%)")

    # CRITICAL CHECK — flag imbalance
    bull_pct = np.mean(val_labels_ssl == 2) * 100
    if bull_pct > 60:
        print(f"\n  ⚠️  WARNING: Bull is {bull_pct:.1f}% of val — probe may collapse to majority class")

    # ═══════════════════════════════════════════
    # CREATE DATALOADERS — needed by all phases
    # ═══════════════════════════════════════════
    ssl_loaders  = create_dataloaders(data['daily_returns'], batch_size=64, mode='full')
    jepa_loaders = create_dataloaders(data['daily_returns'], batch_size=64, mode='jepa')
        

    # ═══════════════════════════════════════════
    # PHASE 3: Supervised Baseline
    # ═══════════════════════════════════════════
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
    sup_val_reprs  = extract_sup_on_ssl_windows(sup_model, ssl_loaders['val'],  device)
    sup_test_reprs = extract_sup_on_ssl_windows(sup_model, ssl_loaders['test'], device)

    print("\n" + "="*60)
    print("PHASE 3 SUMMARY — Supervised")
    print("="*60)
    print(f"  Train dataset: {len(train_sup_ds)} windows")
    print(f"  Val dataset:   {len(val_sup_ds)} windows")
    print(f"  Test dataset:  {len(test_sup_ds)} windows")
    print(f"  Val reprs:  {sup_val_reprs.shape}  ← SSL-aligned (38 windows)")
    print(f"  Test reprs: {sup_test_reprs.shape}  ← SSL-aligned (137 windows)")
    print(f"  Val labels:    {val_labels_sup.shape} — dist: {dict(zip(*np.unique(val_labels_sup, return_counts=True)))}")
    print(f"  Test labels:   {test_labels_sup.shape} — dist: {dict(zip(*np.unique(test_labels_sup, return_counts=True)))}")

    # ═══════════════════════════════════════════
    # PHASE 4: TS2Vec
    # ═══════════════════════════════════════════
    ts2vec_model = train_ts2vec(ssl_loaders['train'], n_epochs=CONFIG['ts2vec_epochs'], device=device)
    ts2vec_val_reprs = extract_ts2vec_representations(ts2vec_model, ssl_loaders['val'], device=device)
    ts2vec_test_reprs = extract_ts2vec_representations(ts2vec_model, ssl_loaders['test'], device=device)

    print("\n" + "="*60)
    print("PHASE 4 SUMMARY — TS2Vec")
    print("="*60)
    print(f"  Val reprs:  {ts2vec_val_reprs.shape}")
    print(f"  Test reprs: {ts2vec_test_reprs.shape}")
    print(f"  Val labels being used:  {val_labels_ssl.shape}")
    print(f"  Test labels being used: {test_labels_ssl.shape}")
    n_match_v = min(len(ts2vec_val_reprs), len(val_labels_ssl))
    n_match_t = min(len(ts2vec_test_reprs), len(test_labels_ssl))
    print(f"  Aligned val:  {n_match_v} pairs")
    print(f"  Aligned test: {n_match_t} pairs")


    # ═══════════════════════════════════════════
    # PHASE 5: PatchTST
    # ═══════════════════════════════════════════
    patchtst_model = train_patchtst(ssl_loaders['train'], n_epochs=CONFIG['patchtst_epochs'], device=device)
    patchtst_val_reprs = extract_patchtst_representations(patchtst_model, ssl_loaders['val'], device=device)
    patchtst_test_reprs = extract_patchtst_representations(patchtst_model, ssl_loaders['test'], device=device)

    print("\n" + "="*60)
    print("PHASE 5 SUMMARY — PatchTST")
    print("="*60)
    print(f"  Val reprs:  {patchtst_val_reprs.shape}")
    print(f"  Test reprs: {patchtst_test_reprs.shape}")
    n_match_v = min(len(patchtst_val_reprs), len(val_labels_ssl))
    n_match_t = min(len(patchtst_test_reprs), len(test_labels_ssl))
    print(f"  Aligned val:  {n_match_v} pairs")
    print(f"  Aligned test: {n_match_t} pairs")

    # ═══════════════════════════════════════════
    # PHASE 6: FinJEPA
    # ═══════════════════════════════════════════
    finjepa_model = train_finjepa(jepa_loaders['train'], n_epochs=CONFIG['finjepa_epochs'], device=device)
    finjepa_val_reprs = extract_finjepa_representations(finjepa_model, jepa_loaders['val'], device=device)
    finjepa_test_reprs = extract_finjepa_representations(finjepa_model, jepa_loaders['test'], device=device)


    # Ablation A1: FinJEPA with fixed tau
    finjepa_fixed = train_finjepa(
        jepa_loaders['train'],
        n_epochs=CONFIG['finjepa_epochs'],
        device=device,
        tau_min=0.996,
        tau_max=0.996
    )
    finjepa_fixed_val  = finjepa_fixed.encode(jepa_loaders['val'])
    finjepa_fixed_test = finjepa_fixed.encode(jepa_loaders['test'])

    print("\n" + "="*60)
    print("PHASE 6 SUMMARY — FinJEPA")
    print("="*60)
    print(f"  Adaptive τ final loss: {0:.6f}")  # replace with actual
    print(f"  Val reprs:  {finjepa_val_reprs.shape}")
    print(f"  Test reprs: {finjepa_test_reprs.shape}")
    print(f"  Val labels being used:  {val_labels_jepa.shape}")
    print(f"  Test labels being used: {test_labels_jepa.shape}")
    n_match_v = min(len(finjepa_val_reprs), len(val_labels_jepa))
    n_match_t = min(len(finjepa_test_reprs), len(test_labels_jepa))
    print(f"  Aligned val:  {n_match_v} pairs")
    print(f"  Aligned test: {n_match_t} pairs")
    if n_match_v != len(val_labels_ssl):
        print(f"  ⚠️  WARNING: JEPA val pairs ({n_match_v}) != SSL val pairs ({len(val_labels_ssl)}) — check stride alignment")
        
    # ═══════════════════════════════════════════
    # PHASE 7: EVALUATION
    # ═══════════════════════════════════════════

    print("\n" + "="*60)
    print("PRE-EVALUATION ALIGNMENT CHECK")
    print("="*60)
    models_check = [
        ("Supervised", sup_val_reprs,      sup_test_reprs,      val_labels_ssl,  test_labels_ssl),
        ("TS2Vec",     ts2vec_val_reprs,   ts2vec_test_reprs,   val_labels_ssl,  test_labels_ssl),
        ("PatchTST",   patchtst_val_reprs, patchtst_test_reprs, val_labels_ssl,  test_labels_ssl),
        ("FinJEPA",    finjepa_val_reprs,  finjepa_test_reprs,  val_labels_jepa, test_labels_jepa),
    ]
    print(f"  {'Model':<20} {'Val reprs':>12} {'Val labels':>12} {'Test reprs':>12} {'Test labels':>12} {'Match?':>8}")
    print("  " + "-"*70)
    for name, vr, tr, vl, tl in models_check:
        val_ok  = "✅" if len(vr) == len(vl) else f"⚠️ {len(vr)} vs {len(vl)}"
        test_ok = "✅" if len(tr) == len(tl) else f"⚠️ {len(tr)} vs {len(tl)}"
        print(f"  {name:<20} {len(vr):>12} {len(vl):>12} {len(tr):>12} {len(tl):>12} {val_ok} / {test_ok}")

    # Check for class imbalance in each label set
    print(f"\n  Label distributions:")
    for name, labels, split in [
        ("SSL val",   val_labels_ssl,  None),
        ("SSL test",  test_labels_ssl, None),
        ("JEPA val",  val_labels_jepa, None),
        ("JEPA test", test_labels_jepa, None),
        ("Sup val",   val_labels_sup,  None),
        ("Sup test",  test_labels_sup, None),
    ]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = {['Bear','Sideways','Bull'][u]: f"{c/len(labels)*100:.0f}%" 
                for u, c in zip(unique, counts)}
        majority_pct = max(counts) / len(labels) * 100
        flag = " ⚠️ IMBALANCED" if majority_pct > 70 else ""
        print(f"    {name:<12}: {dist}{flag}")
        
    all_results = []

    all_results.append(evaluate_random_baseline(test_labels_ssl))
    
    n_sv = min(len(sup_val_reprs), len(val_labels_ssl))
    n_st = min(len(sup_test_reprs), len(test_labels_ssl))
    r = evaluate_model('Supervised',
        sup_val_reprs[:n_sv], sup_test_reprs[:n_st],
        val_labels_ssl[:n_sv], test_labels_ssl[:n_st])
    r['test_labels'] = test_labels_ssl[:n_st]
    all_results.append(r)

    n_tv = min(len(ts2vec_val_reprs),  len(val_labels_ssl))
    n_tt = min(len(ts2vec_test_reprs), len(test_labels_ssl))
    r = evaluate_model('TS2Vec', ts2vec_val_reprs[:n_tv], ts2vec_test_reprs[:n_tt], val_labels_ssl[:n_tv], test_labels_ssl[:n_tt])
    r['test_labels'] = test_labels_ssl[:n_tt]
    all_results.append(r)

    n_pv = min(len(patchtst_val_reprs),  len(val_labels_ssl))
    n_pt = min(len(patchtst_test_reprs), len(test_labels_ssl))
    r = evaluate_model('PatchTST', patchtst_val_reprs[:n_pv], patchtst_test_reprs[:n_pt], val_labels_ssl[:n_pv], test_labels_ssl[:n_pt])
    r['test_labels'] = test_labels_ssl[:n_pt]
    all_results.append(r)

    n_fv = min(len(finjepa_val_reprs),  len(val_labels_jepa))
    n_ft = min(len(finjepa_test_reprs), len(test_labels_jepa))
    r = evaluate_model('FinJEPA', finjepa_val_reprs[:n_fv], finjepa_test_reprs[:n_ft], val_labels_jepa[:n_fv], test_labels_jepa[:n_ft])
    r['test_labels'] = test_labels_jepa[:n_ft]
    all_results.append(r)

    n_fxv = min(len(finjepa_fixed_val),  len(val_labels_jepa))
    n_fxt = min(len(finjepa_fixed_test), len(test_labels_jepa))
    r = evaluate_model(
        'FinJEPA (fixed τ)',
        finjepa_fixed_val[:n_fxv],  finjepa_fixed_test[:n_fxt],
        val_labels_jepa[:n_fxv],    test_labels_jepa[:n_fxt]
    )
    r['test_labels'] = test_labels_jepa[:n_fxt]
    all_results.append(r)

    results_df = generate_results_table(all_results)
    plot_umap_comparison(all_results, test_labels_ssl, save_path=results_dir / "umap_comparison.png")
    plot_confusion_matrices(all_results, save_path=results_dir / "confusion_matrices.png")

    # ═══════════════════════════════════════════
    # PHASE 8: Layer-Wise Probing
    # ═══════════════════════════════════════════
    layerwise_results = evaluate_layerwise_comparison(
        finjepa_model=finjepa_model,
        patchtst_model=patchtst_model,
        val_loader=ssl_loaders['val'],
        test_loader=ssl_loaders['test'],
        val_labels=val_labels_ssl,
        test_labels=test_labels_ssl
    )

    for model_name, layer_results in layerwise_results.items():
        pd.DataFrame(layer_results).to_csv(results_dir / f"layerwise_{model_name.lower()}.csv", index=False)

    # ═══════════════════════════════════════════
    # PHASE 9: Emergence Plot
    # ═══════════════════════════════════════════
    fig = plot_emergence(
        finjepa_results=layerwise_results['FinJEPA'],
        patchtst_results=layerwise_results['PatchTST'],
        save_path=results_dir / "emergence_plot.pdf",
        show=False,
    )
    fig.savefig(results_dir / "emergence_plot.png", dpi=200, bbox_inches='tight', facecolor="#FFFBF5")

    # Save data for paper_figures.py
    np.savez(
        results_dir / "all_representations.npz",
        sup_val=sup_val_reprs, sup_test=sup_test_reprs,
        ts2vec_val=ts2vec_val_reprs, ts2vec_test=ts2vec_test_reprs,
        patchtst_val=patchtst_val_reprs, patchtst_test=patchtst_test_reprs,
        finjepa_val=finjepa_val_reprs, finjepa_test=finjepa_test_reprs,
    )

    # Save confusion matrices for paper_figures.py
    cm_dict = {}
    for r in all_results:
        if 'confusion_matrix' in r:
            cm_dict[f"{r['model'].lower()}_cm"] = r['confusion_matrix']
    np.savez(results_dir / "confusion_matrices.npz", **cm_dict)

    results_df.to_csv(results_dir / "results_table.csv", index=False)

    print("\n✅ Done!")
    return all_results, results_df, layerwise_results

if __name__ == "__main__":
    main()
