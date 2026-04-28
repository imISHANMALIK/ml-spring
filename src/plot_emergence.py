"""
plot_emergence.py — Layer-Wise Regime Emergence Visualization
==============================================================
Visualizes the layer-by-layer growth of linear-probe Macro-F1 for
FinJEPA vs PatchTST, making the central thesis of the paper visible:

  FinJEPA (JEPA, latent-space prediction):
      F1 should climb steeply — abstract regime structure emerges.

  PatchTST (masked reconstruction, input-space prediction):
      F1 should plateau or lag — noise retention limits abstraction.

Aesthetic: minimalist "summer" — warm cream background, coral-orange for
FinJEPA, deep teal for PatchTST, hairline grid, minimal spines.
Publication-ready at 200 dpi; save as .pdf for vector output.

Usage:
    from src.plot_emergence import plot_emergence
    fig = plot_emergence(finjepa_results, patchtst_results,
                         save_path="results/emergence.pdf")
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Use a non-interactive backend when running headless (Colab, server)
try:
    matplotlib.use('Agg')
except Exception:
    pass


# ── Colour palette ────────────────────────────────────────────────────────────
_BG           = "#FFFBF5"   # warm cream — evokes sunlit paper
_FINJEPA_C    = "#E07A5F"   # terracotta-coral (warm, vibrant, summer)
_PATCHTST_C   = "#3D9994"   # deep teal (cool, calm, unambiguous contrast)
_GRID_C       = "#E2D9CE"   # muted warm gray grid lines
_SPINE_C      = "#C8BCAE"   # softened spine / tick color
_TEXT_C       = "#2C2A27"   # near-black body text
_SUBTITLE_C   = "#7A7068"   # muted gray for secondary annotation
_ANNOT_ALPHA  = 0.08        # fill-under-curve alpha


# ── Style helper ─────────────────────────────────────────────────────────────

def _summer_style(ax: plt.Axes, fig: plt.Figure) -> None:
    """Apply the summer minimalist aesthetic in-place."""
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    # Remove top/right spines; mute the remaining two
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(_SPINE_C)
        ax.spines[spine].set_linewidth(0.75)

    # Horizontal-only hairline grid (dotted, very light)
    ax.yaxis.grid(True, color=_GRID_C, linestyle=':', linewidth=0.65, alpha=0.9)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Ticks: small and unobtrusive
    ax.tick_params(axis='both', colors=_TEXT_C, length=3,
                   width=0.65, labelsize=10.5)


# ── Main plot function ────────────────────────────────────────────────────────

def plot_emergence(
    finjepa_results:  list,
    patchtst_results: list,
    save_path=None,
    title:    str = "Market Regime Encoding: Layer-Wise Emergence of Abstract Structure",
    subtitle: str = "Linear probe Macro-F1 on HMM-labeled S&P 500 regimes  ·  frozen encoder  ·  6-layer Transformer",
    figsize:  tuple = (9.2, 5.6),
    show:     bool  = True,
) -> plt.Figure:
    """Plot layer-wise linear-probe Macro-F1 for FinJEPA vs PatchTST.

    Args:
        finjepa_results:  List of dicts with keys 'layer', 'f1', 'accuracy'
                          (returned by evaluate.probe_layerwise).
        patchtst_results: Same format.
        save_path:        Path to save (PDF recommended for publication).
        title:            Main figure title.
        subtitle:         Smaller annotation below the title.
        figsize:          Figure dimensions in inches.
        show:             Call plt.show() at the end (set False in headless envs).

    Returns:
        matplotlib Figure object.
    """
    layers   = [r['layer'] for r in finjepa_results]
    f1_jepa  = [r['f1']    for r in finjepa_results]
    f1_patch = [r['f1']    for r in patchtst_results]

    fig, ax = plt.subplots(figsize=figsize)
    _summer_style(ax, fig)

    # ── Lines ────────────────────────────────────────────────────────────────

    # FinJEPA — coral, filled circles
    ax.plot(
        layers, f1_jepa,
        color=_FINJEPA_C, linewidth=2.5, linestyle='-', zorder=5,
        marker='o', markersize=7.5,
        markerfacecolor=_FINJEPA_C, markeredgecolor='white', markeredgewidth=1.3,
        label='FinJEPA  (JEPA, latent-space prediction)',
    )

    # PatchTST — teal, filled squares
    ax.plot(
        layers, f1_patch,
        color=_PATCHTST_C, linewidth=2.5, linestyle='-', zorder=5,
        marker='s', markersize=7.0,
        markerfacecolor=_PATCHTST_C, markeredgecolor='white', markeredgewidth=1.3,
        label='PatchTST  (masked reconstruction, input-space prediction)',
    )

    # Soft fill under each line — draws the eye without adding clutter
    ax.fill_between(layers, f1_jepa,  alpha=_ANNOT_ALPHA, color=_FINJEPA_C,  zorder=2)
    ax.fill_between(layers, f1_patch, alpha=_ANNOT_ALPHA, color=_PATCHTST_C, zorder=2)

    # ── Data labels at final layer ────────────────────────────────────────────
    _label_final_point(ax, layers[-1], f1_jepa[-1],  _FINJEPA_C,  side='right')
    _label_final_point(ax, layers[-1], f1_patch[-1], _PATCHTST_C, side='right')

    # ── Delta annotation at last layer ──────────────────────────────────────
    delta = f1_jepa[-1] - f1_patch[-1]
    if abs(delta) > 0.005:
        mid_y = (f1_jepa[-1] + f1_patch[-1]) / 2
        sign  = '+' if delta > 0 else ''
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(layers[-1], mid_y),
            xytext=(layers[-1] + 0.15, mid_y),
            fontsize=8.5, color=_SUBTITLE_C,
            va='center', style='italic',
        )

    # ── Axes ────────────────────────────────────────────────────────────────
    ax.set_xlabel(
        "Transformer Layer Depth (1 – 6)",
        fontsize=12, color=_TEXT_C, labelpad=9, fontweight='medium'
    )
    ax.set_ylabel(
        "Linear Probe Macro-F1 Score",
        fontsize=12, color=_TEXT_C, labelpad=9, fontweight='medium'
    )

    ax.set_xticks(layers)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Dynamic Y-range: give 3 % headroom above and below data
    all_f1 = f1_jepa + f1_patch
    y_pad  = (max(all_f1) - min(all_f1)) * 0.18 + 0.02
    ax.set_ylim(max(0.0, min(all_f1) - y_pad),
                min(1.0, max(all_f1) + y_pad * 1.5))
    ax.set_xlim(layers[0] - 0.35, layers[-1] + 0.55)

    # ── Legend ──────────────────────────────────────────────────────────────
    legend = ax.legend(
        frameon=True, framealpha=0.95,
        edgecolor=_GRID_C, facecolor=_BG,
        fontsize=10, loc='upper left',
        handlelength=2.0, handleheight=0.85,
        borderpad=0.8, labelspacing=0.6,
    )
    for text in legend.get_texts():
        text.set_color(_TEXT_C)
    legend.get_frame().set_linewidth(0.6)

    # ── Titles ──────────────────────────────────────────────────────────────
    # Two-line heading: bold main title + italic subtitle via fig.text
    ax.set_title(
        title,
        fontsize=13, fontweight='bold', color=_TEXT_C, pad=22,
    )
    fig.text(
        0.5, 0.955, subtitle,
        ha='center', va='top',
        fontsize=9.5, color=_SUBTITLE_C, style='italic'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # ── Save ────────────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = save_path.suffix.lstrip('.') or 'pdf'
        fig.savefig(
            save_path, dpi=200, bbox_inches='tight',
            facecolor=_BG, format=fmt
        )
        print(f"Saved emergence plot → {save_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass  # headless environment

    return fig


# ── Annotation helper ─────────────────────────────────────────────────────────

def _label_final_point(ax, x, y, color, side='right', offset=0.08):
    """Place a small numeric label next to the final data point."""
    x_off = offset if side == 'right' else -offset
    ax.annotate(
        f"{y:.3f}",
        xy=(x, y), xytext=(x + x_off, y),
        fontsize=9, color=color, fontweight='bold',
        va='center', ha='left' if side == 'right' else 'right',
    )


# ── Optional: multi-run confidence band version ──────────────────────────────

def plot_emergence_with_bands(
    finjepa_runs:  list,   # list of result-lists (one per run)
    patchtst_runs: list,
    save_path=None,
    **kwargs,
) -> plt.Figure:
    """Like plot_emergence but with mean ± std bands across multiple runs.

    Args:
        finjepa_runs:  List of result lists (each from probe_layerwise)
        patchtst_runs: Same format

    Each result list is the output of evaluate.probe_layerwise() for one run.
    """
    def _aggregate(runs):
        """Returns (layers, mean_f1, std_f1)."""
        layers  = [r['layer'] for r in runs[0]]
        all_f1  = np.array([[r['f1'] for r in run] for run in runs])
        return layers, all_f1.mean(axis=0), all_f1.std(axis=0)

    layers,  fj_mean, fj_std  = _aggregate(finjepa_runs)
    _,       pt_mean, pt_std  = _aggregate(patchtst_runs)

    # Build synthetic single-run dicts for the base plot
    fj_single = [{'layer': l, 'f1': m, 'accuracy': 0} for l, m in zip(layers, fj_mean)]
    pt_single = [{'layer': l, 'f1': m, 'accuracy': 0} for l, m in zip(layers, pt_mean)]

    fig = plot_emergence(fj_single, pt_single, show=False, **kwargs)
    ax  = fig.axes[0]

    # Overlay shaded bands
    ax.fill_between(layers, fj_mean - fj_std, fj_mean + fj_std,
                    alpha=0.15, color=_FINJEPA_C, zorder=3)
    ax.fill_between(layers, pt_mean - pt_std, pt_mean + pt_std,
                    alpha=0.15, color=_PATCHTST_C, zorder=3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = save_path.suffix.lstrip('.') or 'pdf'
        fig.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor=_BG, format=fmt)
        print(f"Saved emergence (with bands) → {save_path}")

    try:
        plt.show()
    except Exception:
        pass

    return fig
