"""
FinJEPA EMA Decay (tau) Hyperparameter Sweep
==============================================
Runs FinJEPA on the Dense Sliding Window dataset with different
adaptive and fixed EMA configurations to mathematically justify
the architectural choices.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

from src.data_pipeline import load_and_preprocess, create_dataloaders, DenseRegimeDataset
from src.hmm_labels import RegimeLabeler
from src.finjepa import train_finjepa, extract_finjepa_representations
from src.supervised_baseline import SupervisedBaseline, extract_representations
from src.evaluate import evaluate_all_models

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Data
    data = load_and_preprocess()
    jepa_loaders = create_dataloaders(data['daily_returns'], batch_size=64, mode='jepa')
    
    # 2. Get HMM Labels for evaluation
    labeler = RegimeLabeler(n_states=3)
    train_returns = data['daily_returns']['train']['z_return'].values
    labeler.fit(train_returns)
    daily_labels = {}
    for split in ['val', 'test']:
        daily_labels[split] = labeler.predict(data['daily_returns'][split]['z_return'].values)
    
    # Extract ground truth labels for the exact patch indices (using Supervised dataset dummy run)
    val_sup_ds = DenseRegimeDataset(data['daily_returns']['val']['z_return'].values, daily_labels['val'])
    test_sup_ds = DenseRegimeDataset(data['daily_returns']['test']['z_return'].values, daily_labels['test'])
    dummy_model = SupervisedBaseline(patch_size=20)
    _, val_labels_ext = extract_representations(dummy_model, val_sup_ds, device=device)
    _, test_labels_ext = extract_representations(dummy_model, test_sup_ds, device=device)
    
    labels_dict = {'patch_labels': {'val': val_labels_ext, 'test': test_labels_ext}}
    
    # 3. Define the Sweep Configurations
    # Format: "Name": (tau_min, tau_max)
    configs = {
        "Fixed (Fast)": (0.950, 0.950),
        "Fixed (Baseline)": (0.996, 0.996), # Standard I-JEPA
        "Frozen Target": (1.000, 1.000),
        "Adaptive (Aggressive)": (0.950, 0.999),
        "Adaptive (Proposed)": (0.990, 0.999), # Our proposed FinJEPA
    }
    
    sweep_results = []
    
    for name, (t_min, t_max) in configs.items():
        print(f"\n{'='*50}\nTesting Configuration: {name} (tau_min={t_min}, tau_max={t_max})\n{'='*50}")
        
        # Train FinJEPA
        model = train_finjepa(
            jepa_loaders['train'], 
            d_model=384, 
            n_epochs=100, # Using 100 for sweep speed
            device=device,
            tau_min=t_min,
            tau_max=t_max
        )
        
        # Extract Representations
        val_reprs = extract_finjepa_representations(model, jepa_loaders['val'], device=device)
        test_reprs = extract_finjepa_representations(model, jepa_loaders['test'], device=device)
        
        # Evaluate
        representations = {
            name: {'val': val_reprs, 'test': test_reprs}
        }
        
        res, _ = evaluate_all_models(representations, labels_dict, save_dir=Path('results/sweep'))
        
        f1_score = res[name]['regime_f1_macro']
        sil_score = res[name]['silhouette']
        
        sweep_results.append({
            "Configuration": name,
            "Tau Min": t_min,
            "Tau Max": t_max,
            "Macro F1": f1_score,
            "Silhouette": sil_score
        })
        print(f"--> Result: F1={f1_score:.4f}, Silhouette={sil_score:.4f}")

    # 4. Save and Print Results
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
