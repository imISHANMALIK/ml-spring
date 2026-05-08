"""
FinJEPA EMA Decay (tau) Hyperparameter Sweep
==============================================
Runs FinJEPA on the Dense Sliding Window dataset with different
adaptive and fixed EMA configurations to mathematically justify
the architectural choices.

Updated to align with Mehul's fixes:
  - HMM fitted on log_return (not z_return)
  - Proper JEPA label alignment (label = last day of context, not full window)
  - class_weight='balanced' in probes (handled in evaluate.py)
  - Global seed for reproducibility
  - Uses evaluate_model() directly with per-config aligned labels
"""

import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
import sys

from src.data_pipeline import (load_and_preprocess, create_dataloaders,
                                PATCH_SIZE, CONTEXT_PATCHES, TARGET_PATCHES, STRIDE)
from src.hmm_labels import RegimeLabeler
from src.finjepa import train_finjepa, extract_finjepa_representations
from src.evaluate import evaluate_model


def main():
    # ── Global seed for reproducibility ──────────────────────────
    SEED = 12
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {SEED}")

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ── 1. Load Data ─────────────────────────────────────────────
    data = load_and_preprocess()
    jepa_loaders = create_dataloaders(data['daily_returns'], batch_size=64, mode='jepa')
    
    # ── 2. HMM Labels — fitted on log_return (Mehul's fix) ──────
    labeler = RegimeLabeler(n_states=3)
    train_returns = data['daily_returns']['train']['log_return'].values
    labeler.fit(train_returns)

    daily_labels = {}
    for split in ['train', 'val', 'test']:
        daily_labels[split] = labeler.predict(
            data['daily_returns'][split]['log_return'].values
        )

    # ── 3. JEPA-aligned labels ──────────────────────────────────
    # FinJEPA uses 320-day windows (context=240 + target=80) but the
    # representation comes from the context encoder only. So the label
    # should correspond to the last day of the CONTEXT window (day 240),
    # not the last day of the full window (day 320).
    def get_jepa_labels(daily_labels_arr, stride=STRIDE):
        """Label = last day of context window (day 240), not day 320."""
        total_window = (CONTEXT_PATCHES + TARGET_PATCHES) * PATCH_SIZE  # 320
        context_end  = CONTEXT_PATCHES * PATCH_SIZE                      # 240
        n = max(0, (len(daily_labels_arr) - total_window) // stride + 1)
        return np.array([
            daily_labels_arr[i * stride + context_end - 1]
            for i in range(n)
        ])

    val_labels_jepa  = get_jepa_labels(daily_labels['val'])
    test_labels_jepa = get_jepa_labels(daily_labels['test'])

    print(f"\nJEPA label counts — val: {len(val_labels_jepa)}, test: {len(test_labels_jepa)}")
    for name, labels in [("Val", val_labels_jepa), ("Test", test_labels_jepa)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist = {['Bear','Sideways','Bull'][u]: f"{c/len(labels)*100:.0f}%"
                for u, c in zip(unique, counts)}
        print(f"  {name}: {dist}")

    # ── 4. Define Sweep Configurations ──────────────────────────
    # Format: "Name": (tau_min, tau_max)
    configs = {
        "Fixed (Fast)":           (0.950, 0.950),
        "Fixed (Baseline)":       (0.996, 0.996),   # Standard I-JEPA
        "Frozen Target":          (1.000, 1.000),
        "Adaptive (Aggressive)":  (0.950, 0.999),
        "Adaptive (Proposed)":    (0.990, 0.999),   # Our proposed FinJEPA
    }
    
    sweep_results = []
    
    for name, (t_min, t_max) in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing Configuration: {name} (tau_min={t_min}, tau_max={t_max})")
        print(f"{'='*60}")
        
        # Train FinJEPA
        model = train_finjepa(
            jepa_loaders['train'], 
            d_model=384, 
            n_epochs=100,  # 100 for sweep speed (vs 400 for full run)
            device=device,
            tau_min=t_min,
            tau_max=t_max
        )
        
        # Extract Representations
        val_reprs  = extract_finjepa_representations(model, jepa_loaders['val'],  device=device)
        test_reprs = extract_finjepa_representations(model, jepa_loaders['test'], device=device)
        
        # Align repr count with label count
        n_val  = min(len(val_reprs),  len(val_labels_jepa))
        n_test = min(len(test_reprs), len(test_labels_jepa))
        
        print(f"  Repr alignment: val={len(val_reprs)} reprs / {len(val_labels_jepa)} labels → {n_val} pairs")
        print(f"  Repr alignment: test={len(test_reprs)} reprs / {len(test_labels_jepa)} labels → {n_test} pairs")
        
        # Evaluate with aligned labels (class_weight='balanced' is in evaluate_model)
        result = evaluate_model(
            name,
            val_reprs[:n_val],          test_reprs[:n_test],
            val_labels_jepa[:n_val],    test_labels_jepa[:n_test]
        )
        
        f1_score  = result['regime_f1']
        sil_score = result.get('silhouette')
        
        sweep_results.append({
            "Configuration": name,
            "Tau Min": t_min,
            "Tau Max": t_max,
            "Macro F1": f1_score,
            "Accuracy": result['regime_accuracy'],
            "Silhouette": sil_score if sil_score is not None else float('nan'),
        })
        print(f"--> Result: F1={f1_score:.4f}, "
              f"Acc={result['regime_accuracy']:.4f}, "
              f"Sil={sil_score:.4f}" if sil_score else f"--> Result: F1={f1_score:.4f}")

    # ── 5. Save and Print Results ────────────────────────────────
    df_results = pd.DataFrame(sweep_results)
    print("\n\n" + "="*60)
    print("HYPERPARAMETER SWEEP RESULTS (Adaptive EMA τ)")
    print("="*60)
    print(df_results.to_string(index=False))
    
    Path('results/sweep').mkdir(parents=True, exist_ok=True)
    df_results.to_csv('results/sweep/tau_sweep_results.csv', index=False)
    print("\nResults saved to results/sweep/tau_sweep_results.csv")

if __name__ == "__main__":
    main()
