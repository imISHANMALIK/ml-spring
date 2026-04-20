"""
Unified Evaluation (Phase 6)
==============================
All models are evaluated identically:
    1. Freeze encoder → extract representations
    2. Train linear probe on val set (using HMM labels)
    3. Evaluate on test set (using HMM labels)

Metrics:
    - Macro-F1 for regime classification
    - Silhouette score for UMAP clustering
    - Return forecasting MSE (5-day forward return)
    - Sharpe ratio for simple trading strategy

Usage:
    from src.evaluate import evaluate_all_models, generate_results_table
    results = evaluate_all_models(representations_dict, labels_data)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (f1_score, classification_report, 
                             confusion_matrix, silhouette_score,
                             mean_squared_error)
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. UMAP visualizations will be skipped.")


# ─────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────
def evaluate_model(model_name, val_reprs, test_reprs, 
                   val_labels, test_labels,
                   val_returns=None, test_returns=None):
    """Evaluate a single model with the unified protocol.
    
    Args:
        model_name: str
        val_reprs: np.array (n_val, d_model) — representations on val set
        test_reprs: np.array (n_test, d_model) — representations on test set
        val_labels: np.array (n_val,) — HMM regime labels for val
        test_labels: np.array (n_test,) — HMM regime labels for test
        val_returns: optional np.array — for return forecasting
        test_returns: optional np.array — for return forecasting
        
    Returns:
        dict with metrics
    """
    results = {'model': model_name}
    
    # --- Regime Classification (main metric) ---
    # Normalize representations
    scaler = StandardScaler()
    val_reprs_scaled = scaler.fit_transform(val_reprs)
    test_reprs_scaled = scaler.transform(test_reprs)
    
    # Train linear probe on val
    probe = LogisticRegression(
        max_iter=2000, 
        C=1.0,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )
    probe.fit(val_reprs_scaled, val_labels)
    
    # Predict on test
    test_preds = probe.predict(test_reprs_scaled)
    
    # Metrics
    results['regime_f1'] = f1_score(test_labels, test_preds, average='macro')
    results['regime_accuracy'] = np.mean(test_preds == test_labels)
    results['confusion_matrix'] = confusion_matrix(test_labels, test_preds)
    results['classification_report'] = classification_report(
        test_labels, test_preds, 
        target_names=['Bear', 'Sideways', 'Bull'],
        output_dict=True
    )
    
    # UMAP clustering quality
    if HAS_UMAP and len(test_reprs) > 10:
        try:
            umap_embed = UMAP(n_components=2, random_state=42, 
                             n_neighbors=min(15, len(test_reprs)-1)).fit_transform(test_reprs_scaled)
            results['silhouette'] = silhouette_score(umap_embed, test_labels)
            results['umap_embedding'] = umap_embed
        except Exception as e:
            print(f"  UMAP failed for {model_name}: {e}")
            results['silhouette'] = None
            results['umap_embedding'] = None
    else:
        results['silhouette'] = None
        results['umap_embedding'] = None
    
    # --- Return Forecasting (secondary metric) ---
    if val_returns is not None and test_returns is not None:
        ridge = Ridge(alpha=1.0)
        ridge.fit(val_reprs_scaled, val_returns)
        return_preds = ridge.predict(test_reprs_scaled)
        results['forecast_mse'] = mean_squared_error(test_returns, return_preds)
        results['return_predictions'] = return_preds
    else:
        results['forecast_mse'] = None
    
    # --- Trading Strategy (Sharpe Ratio) ---
    if val_returns is not None and test_returns is not None:
        # Simple strategy: buy if predicted 5-day return > 0, sell after 5 days
        signals = (return_preds > 0).astype(float)   # 1 = long, 0 = cash
        strategy_returns = signals * test_returns
        
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        results['sharpe_ratio'] = sharpe
    else:
        results['sharpe_ratio'] = None
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"  Regime F1 (macro): {results['regime_f1']:.4f}")
    print(f"  Regime Accuracy:   {results['regime_accuracy']:.4f}")
    if results['silhouette'] is not None:
        print(f"  Silhouette Score:  {results['silhouette']:.4f}")
    if results['forecast_mse'] is not None:
        print(f"  Forecast MSE:      {results['forecast_mse']:.6f}")
    if results['sharpe_ratio'] is not None:
        print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.4f}")
    
    return results


def evaluate_random_baseline(test_labels, n_classes=3):
    """Random classifier baseline — always predicts random class."""
    results = {'model': 'Random'}
    
    np.random.seed(42)
    preds = np.random.randint(0, n_classes, size=len(test_labels))
    
    results['regime_f1'] = f1_score(test_labels, preds, average='macro')
    results['regime_accuracy'] = np.mean(preds == test_labels)
    results['confusion_matrix'] = confusion_matrix(test_labels, preds)
    results['silhouette'] = None
    results['forecast_mse'] = None
    results['sharpe_ratio'] = 0.0
    
    print(f"\n{'='*50}")
    print(f"Model: Random Baseline")
    print(f"{'='*50}")
    print(f"  Regime F1 (macro): {results['regime_f1']:.4f}")
    
    return results


# ─────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────
def generate_results_table(all_results):
    """Generate the final results table (matches flowchart 2).
    
    Returns:
        DataFrame with columns: Model, Regime F1, Forecast MSE, Sharpe, Labels used?
    """
    rows = []
    for r in all_results:
        rows.append({
            'Model': r['model'],
            'Regime F1': f"{r['regime_f1']:.4f}" if r['regime_f1'] is not None else '—',
            'Forecast MSE': f"{r['forecast_mse']:.6f}" if r.get('forecast_mse') else '—',
            'Sharpe': f"{r['sharpe_ratio']:.2f}" if r.get('sharpe_ratio') is not None else '—',
            'Silhouette': f"{r.get('silhouette', 0):.4f}" if r.get('silhouette') is not None else '—',
            'Labels used?': 'Yes' if r['model'] == 'Supervised' else 'No'
        })
    
    df = pd.DataFrame(rows)
    
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def plot_umap_comparison(all_results, test_labels, save_path=None):
    """Plot UMAP embeddings for all models side-by-side.
    
    Color by HMM regime — good models should show clear regime clusters.
    """
    models_with_umap = [r for r in all_results 
                        if r.get('umap_embedding') is not None]
    
    if not models_with_umap:
        print("No UMAP embeddings available. Skipping plot.")
        return
    
    n_models = len(models_with_umap)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    colors = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}
    state_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
    
    for ax, result in zip(axes, models_with_umap):
        embed = result['umap_embedding']
        # Align labels to representation length
        n_reprs = len(embed)
        labels = test_labels[-n_reprs:]  # Use last n labels (aligned with representations)
        
        for regime in [0, 1, 2]:
            mask = labels == regime
            if mask.any():
                ax.scatter(embed[mask, 0], embed[mask, 1],
                          c=colors[regime], s=30, alpha=0.7,
                          label=state_names[regime], edgecolors='white', linewidth=0.3)
        
        sil = result.get('silhouette', 0)
        sil_str = f"{sil:.3f}" if sil is not None else "N/A"
        ax.set_title(f"{result['model']}\n(Silhouette: {sil_str})",
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.15)
    
    fig.patch.set_facecolor('#0f0f1e')
    plt.suptitle('UMAP Visualization of Learned Representations\n'
                 '(colored by HMM regime labels)',
                 fontsize=14, fontweight='bold', color='white')
    
    for ax in axes:
        ax.tick_params(colors='white')
        ax.title.set_color('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   facecolor=fig.get_facecolor())
        print(f"Saved UMAP comparison to {save_path}")
    
    plt.show()
    return fig


def plot_confusion_matrices(all_results, save_path=None):
    """Plot confusion matrices for all models."""
    models_with_cm = [r for r in all_results if 'confusion_matrix' in r]
    
    n_models = len(models_with_cm)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    state_names = ['Bear', 'Sideways', 'Bull']
    
    for ax, result in zip(axes, models_with_cm):
        cm = result['confusion_matrix']
        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=state_names, yticklabels=state_names,
                   ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(f"{result['model']}\nF1={result['regime_f1']:.3f}", fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    plt.suptitle('Confusion Matrices (normalized by row)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrices to {save_path}")
    
    plt.show()
    return fig


# ─────────────────────────────────────────────
# Convenience: evaluate everything
# ─────────────────────────────────────────────
def evaluate_all_models(representations, labels_data, 
                        forward_returns=None, save_dir=None):
    """Evaluate all models and generate results.
    
    Args:
        representations: dict mapping model_name → {
            'val': np.array (n_val, d),
            'test': np.array (n_test, d)
        }
        labels_data: dict from RegimeLabeler.fit_and_label()
        forward_returns: optional dict with 'val' and 'test' forward returns
        save_dir: optional path to save plots and results
        
    Returns:
        all_results: list of result dicts
        results_df: DataFrame with results table
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Random baseline
    test_labels = labels_data['patch_labels']['test']
    random_result = evaluate_random_baseline(test_labels)
    all_results.append(random_result)
    
    # Each model
    for model_name, reprs in representations.items():
        val_reprs = reprs['val']
        test_reprs = reprs['test']
        
        # Align labels with representations
        # (some models produce fewer outputs due to context window requirements)
        val_labels = labels_data['patch_labels']['val']
        n_val = min(len(val_reprs), len(val_labels))
        val_labels_aligned = val_labels[-n_val:]
        val_reprs_aligned = val_reprs[-n_val:]
        
        n_test = min(len(test_reprs), len(test_labels))
        test_labels_aligned = test_labels[-n_test:]
        test_reprs_aligned = test_reprs[-n_test:]
        
        # Forward returns for forecasting/Sharpe
        val_ret = None
        test_ret = None
        if forward_returns:
            val_ret = forward_returns.get('val')
            test_ret = forward_returns.get('test')
            if val_ret is not None:
                val_ret = val_ret[-n_val:]
            if test_ret is not None:
                test_ret = test_ret[-n_test:]
        
        result = evaluate_model(
            model_name,
            val_reprs_aligned, test_reprs_aligned,
            val_labels_aligned, test_labels_aligned,
            val_ret, test_ret
        )
        all_results.append(result)
    
    # Generate results table
    results_df = generate_results_table(all_results)
    
    # Visualizations
    if save_dir:
        results_df.to_csv(save_dir / "results_table.csv", index=False)
        
        plot_umap_comparison(all_results, test_labels, 
                           save_path=save_dir / "umap_comparison.png")
        plot_confusion_matrices(all_results,
                              save_path=save_dir / "confusion_matrices.png")
    
    return all_results, results_df
