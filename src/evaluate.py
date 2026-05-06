"""
Unified Evaluation (Phase 6) + Layer-Wise Probing (Phase 8)
=============================================================
Standard evaluation protocol (unchanged):
    1. Freeze encoder → extract final-layer representations
    2. Train linear probe on val set (using HMM labels)
    3. Evaluate on test set → Macro-F1, Silhouette, Sharpe

Layer-wise probing (new):
    For each transformer layer (1 – 6) of FinJEPA and PatchTST:
        - Extract global-average-pooled hidden state
        - Train a logistic regression probe on val
        - Record Macro-F1 and accuracy on test
    The trajectory of F1 across layers reveals whether abstract regime
    structure *emerges* gradually (FinJEPA) or stagnates due to noise
    overfitting (PatchTST).

Usage:
    from src.evaluate import evaluate_all_models, evaluate_layerwise_comparison
    results = evaluate_all_models(representations_dict, labels_data)
    layerwise = evaluate_layerwise_comparison(finjepa_model, patchtst_model, ...)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (f1_score, classification_report,
                             confusion_matrix, silhouette_score,
                             mean_squared_error)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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
# Core single-model evaluation (unchanged)
# ─────────────────────────────────────────────

def evaluate_model(model_name, val_reprs, test_reprs,
                   val_labels, test_labels,
                   val_returns=None, test_returns=None):
    """Evaluate a single model with the unified protocol.

    Args:
        model_name: str
        val_reprs:  np.array (n_val, d_model)
        test_reprs: np.array (n_test, d_model)
        val_labels: np.array (n_val,)
        test_labels:np.array (n_test,)

    Returns:
        dict with metrics
    """
    results = {'model': model_name}

    scaler = StandardScaler()
    val_scaled  = scaler.fit_transform(val_reprs)
    test_scaled = scaler.transform(test_reprs)

    probe = LogisticRegression(
        max_iter=2000, C=1.0,
        solver='lbfgs', random_state=42,
        class_weight='balanced'
    )
    probe.fit(val_scaled, val_labels)
    test_preds = probe.predict(test_scaled)

    results['regime_f1']       = f1_score(test_labels, test_preds, average='macro')
    results['regime_accuracy'] = float(np.mean(test_preds == test_labels))
    results['confusion_matrix'] = confusion_matrix(test_labels, test_preds)
    results['classification_report'] = classification_report(
        test_labels, test_preds,
        labels=[0, 1, 2],
        target_names=['Bear', 'Sideways', 'Bull'], output_dict=True,
        zero_division=0
    )

    if HAS_UMAP and len(test_reprs) > 10:
        try:
            umap_embed = UMAP(n_components=2, random_state=42,
                             n_neighbors=min(15, len(test_reprs)-1)
                             ).fit_transform(test_scaled)
            results['silhouette']     = silhouette_score(umap_embed, test_labels)
            results['umap_embedding'] = umap_embed
        except Exception as e:
            print(f"  UMAP failed for {model_name}: {e}")
            results['silhouette']     = None
            results['umap_embedding'] = None
    else:
        results['silhouette']     = None
        results['umap_embedding'] = None

    if val_returns is not None and test_returns is not None:
        ridge = Ridge(alpha=1.0)
        ridge.fit(val_scaled, val_returns)
        return_preds = ridge.predict(test_scaled)
        results['forecast_mse']       = mean_squared_error(test_returns, return_preds)
        results['return_predictions'] = return_preds

        signals          = (return_preds > 0).astype(float)
        strategy_returns = signals * test_returns
        results['sharpe_ratio'] = (
            (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
            if strategy_returns.std() > 0 else 0.0
        )
    else:
        results['forecast_mse']  = None
        results['sharpe_ratio']  = None

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
    results = {'model': 'Random'}
    np.random.seed(42)
    preds = np.random.randint(0, n_classes, size=len(test_labels))
    results['regime_f1']       = f1_score(test_labels, preds, average='macro')
    results['regime_accuracy'] = float(np.mean(preds == test_labels))
    results['confusion_matrix'] = confusion_matrix(test_labels, preds)
    results['silhouette']  = None
    results['forecast_mse'] = None
    results['sharpe_ratio'] = 0.0
    print(f"\n{'='*50}\nModel: Random Baseline\n{'='*50}")
    print(f"  Regime F1 (macro): {results['regime_f1']:.4f}")
    return results


# ─────────────────────────────────────────────
# Layer-wise probing  (NEW)
# ─────────────────────────────────────────────

def probe_layerwise(model_name: str,
                    layerwise_val: list,
                    layerwise_test: list,
                    val_labels: np.ndarray,
                    test_labels: np.ndarray) -> list:
    """Train one logistic regression probe per transformer layer.

    Args:
        model_name:     Display name for logging
        layerwise_val:  List of n_layers np.arrays, each (n_val_windows, d_model)
        layerwise_test: List of n_layers np.arrays, each (n_test_windows, d_model)
        val_labels:     HMM labels for the val set   (n_val_patches,)
        test_labels:    HMM labels for the test set  (n_test_patches,)

    Returns:
        List of dicts — one per layer — with keys 'layer', 'f1', 'accuracy'.
    """
    n_layers = len(layerwise_val)
    results  = []

    print(f"\n  Layer-wise probing: {model_name}")
    print(f"  {'Layer':<8} {'Macro-F1':>10} {'Accuracy':>10}")
    print(f"  {'-'*30}")

    for layer_idx in range(n_layers):
        val_reprs  = layerwise_val[layer_idx]   # (n_val_windows, d_model)
        test_reprs = layerwise_test[layer_idx]  # (n_test_windows, d_model)

        # Align: representations may be fewer than patches due to context windowing
        n_val  = min(len(val_reprs),  len(val_labels))
        n_test = min(len(test_reprs), len(test_labels))

        scaler = StandardScaler()
        val_scaled  = scaler.fit_transform(val_reprs[-n_val:])
        test_scaled = scaler.transform(test_reprs[-n_test:])

        val_y  = val_labels[-n_val:]
        test_y = test_labels[-n_test:]

        probe = LogisticRegression(
            max_iter=2000, C=1.0,
            solver='lbfgs', random_state=42,
            class_weight='balanced'
        )
        probe.fit(val_scaled, val_y)
        preds = probe.predict(test_scaled)

        f1  = float(f1_score(test_y, preds, average='macro'))
        acc = float(np.mean(preds == test_y))

        results.append({'layer': layer_idx + 1, 'f1': f1, 'accuracy': acc})
        print(f"  Layer {layer_idx+1:<4}  {f1:>10.4f}  {acc:>10.4f}")

    return results


def evaluate_layerwise_comparison(finjepa_model,
                                  patchtst_model,
                                  val_loader,
                                  test_loader,
                                  val_labels:    np.ndarray,
                                  test_labels:   np.ndarray) -> dict:
    """Run layer-wise probing for FinJEPA and PatchTST and return results."""
    print("\n" + "="*60)
    print("LAYER-WISE PROBING EVALUATION")
    print("="*60)

    print("\nExtracting FinJEPA layerwise representations...")
    fj_val  = finjepa_model.encode_layerwise(val_loader)
    fj_test = finjepa_model.encode_layerwise(test_loader)

    print("\nExtracting PatchTST layerwise representations...")
    pt_val  = patchtst_model.encode_layerwise(val_loader)
    pt_test = patchtst_model.encode_layerwise(test_loader)

    fj_results = probe_layerwise('FinJEPA',  fj_val,  fj_test,  val_labels, test_labels)
    pt_results = probe_layerwise('PatchTST', pt_val,  pt_test,  val_labels, test_labels)

    print("\n" + "="*60)
    print("LAYER-WISE SUMMARY")
    print("="*60)
    print(f"  {'Layer':<8} {'FinJEPA F1':>12} {'PatchTST F1':>13} {'Delta':>8}")
    print(f"  {'-'*45}")
    for fj, pt in zip(fj_results, pt_results):
        delta = fj['f1'] - pt['f1']
        sign  = '+' if delta >= 0 else ''
        print(f"  Layer {fj['layer']:<4}  {fj['f1']:>12.4f}  {pt['f1']:>13.4f}  "
              f"{sign}{delta:>7.4f}")

    return {'FinJEPA': fj_results, 'PatchTST': pt_results}


# ─────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────

def generate_results_table(all_results):
    rows = []
    for r in all_results:
        rows.append({
            'Model':        r['model'],
            'Regime F1':    f"{r['regime_f1']:.4f}" if r['regime_f1'] is not None else '—',
            'Forecast MSE': f"{r['forecast_mse']:.6f}" if r.get('forecast_mse') else '—',
            'Sharpe':       f"{r['sharpe_ratio']:.2f}" if r.get('sharpe_ratio') is not None else '—',
            'Silhouette':   f"{r.get('silhouette', 0):.4f}" if r.get('silhouette') is not None else '—',
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
# Visualization helpers (unchanged)
# ─────────────────────────────────────────────

def plot_umap_comparison(all_results, test_labels, save_path=None):
    models_with_umap = [r for r in all_results if r.get('umap_embedding') is not None]
    if not models_with_umap:
        print("No UMAP embeddings available. Skipping plot.")
        return

    n_models = len(models_with_umap)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]

    colors     = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}
    state_names = {0: 'Bear',  1: 'Sideways', 2: 'Bull'}

    for ax, result in zip(axes, models_with_umap):
        embed  = result['umap_embedding']
        n_repr = len(embed)
        # Use per-model labels stored in result if available, else fall back
        model_test_labels = result.get('test_labels', test_labels)
        labels = model_test_labels[-n_repr:]
        for regime in [0, 1, 2]:
            mask = labels == regime
            if mask.any():
                ax.scatter(embed[mask, 0], embed[mask, 1],
                          c=colors[regime], s=30, alpha=0.7,
                          label=state_names[regime],
                          edgecolors='white', linewidth=0.3)
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
    models_with_cm = [r for r in all_results if 'confusion_matrix' in r]
    n_models = len(models_with_cm)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]

    state_names = ['Bear', 'Sideways', 'Bull']
    for ax, result in zip(axes, models_with_cm):
        cm      = result['confusion_matrix']
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=state_names, yticklabels=state_names,
                   ax=ax, vmin=0, vmax=1, cbar=False)
        ax.set_title(f"{result['model']}\nF1={result['regime_f1']:.3f}",
                    fontweight='bold')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

    plt.suptitle('Confusion Matrices (normalized by row)',
                fontsize=14, fontweight='bold')
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
    """Evaluate all models with the standard final-layer protocol.

    Args:
        representations: dict  model_name → {'val': array, 'test': array}
        labels_data:     dict from RegimeLabeler.fit_and_label()
        forward_returns: optional dict with 'val' and 'test' arrays
        save_dir:        optional path to save plots / CSV

    Returns:
        (all_results, results_df)
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    test_labels = labels_data['patch_labels']['test']

    all_results.append(evaluate_random_baseline(test_labels))

    for model_name, reprs in representations.items():
        val_reprs   = reprs['val']
        test_reprs  = reprs['test']
        val_labels  = labels_data['patch_labels']['val']

        n_val  = min(len(val_reprs),  len(val_labels))
        n_test = min(len(test_reprs), len(test_labels))

        val_ret  = None
        test_ret = None
        if forward_returns:
            val_ret  = forward_returns.get('val')
            test_ret = forward_returns.get('test')
            if val_ret  is not None: val_ret  = val_ret[-n_val:]
            if test_ret is not None: test_ret = test_ret[-n_test:]

        result = evaluate_model(
            model_name,
            val_reprs[-n_val:],   test_reprs[-n_test:],
            val_labels[-n_val:],  test_labels[-n_test:],
            val_ret, test_ret
        )
        all_results.append(result)

    results_df = generate_results_table(all_results)

    if save_dir:
        results_df.to_csv(save_dir / "results_table.csv", index=False)
        plot_umap_comparison(all_results, test_labels,
                            save_path=save_dir / "umap_comparison.png")
        plot_confusion_matrices(all_results,
                               save_path=save_dir / "confusion_matrices.png")

    return all_results, results_df
