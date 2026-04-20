# FinJEPA — Market Regime Detection

Self-supervised representation learning for financial market regime detection using S&P 500 data.

## Project Structure

| Track | Model | Type | Owner |
|-------|-------|------|-------|
| **A** | HMM (3-state) | Ground truth labels | Sahil |
| **B** | FinJEPA | JEPA pretraining (main model) | Mehul |
| **C** | Privileged Learning | TBD | Person 3 |
| **D** | TS2Vec | Contrastive SSL baseline | Sahil |
| **E** | PatchTST | Masked reconstruction baseline | Sahil |
| — | Supervised | End-to-end on HMM labels | Sahil |

## Quick Start (Colab)

```bash
# Cell 1: Install dependencies
!pip install -q yfinance hmmlearn umap-learn

# Cell 2: Paste entire colab_run_all.py and run
```

1. Open [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → **GPU (T4)**
3. Run the install cell above
4. Copy-paste `colab_run_all.py` into a cell and run (~15 min)

## Quick Start (Local)

```bash
pip install -r requirements.txt
python run_all.py
```

## Pipeline

```
S&P 500 (2000-2024)
  → Log returns → Rolling z-score → 20-day patches
  → Train/Val/Test split (2000-2019 / 2020-2021 / 2022-2024)
  
Track A: 3-state HMM → ground truth regime labels (Bear/Sideways/Bull)
  → Exported as .npy for all models

Baselines (all train on TRAIN set, no labels):
  Supervised: 6-layer transformer + cross-entropy on HMM labels
  TS2Vec:     Dilated CNN + contrastive loss
  PatchTST:   Transformer + masked patch reconstruction (MSE)

Evaluation (identical for ALL models):
  Freeze encoder → 384-dim representations
  → Linear probe trained on VAL (HMM labels)
  → Macro-F1 on TEST
  → UMAP visualization + silhouette score
```

## Files

```
├── colab_run_all.py           # ⭐ Single-file Colab script (everything inline)
├── run_all.py                 # Local execution (imports from src/)
├── requirements.txt
├── src/
│   ├── data_pipeline.py       # Data download, preprocessing, patching
│   ├── hmm_labels.py          # Track A: 3-state HMM ground truth
│   ├── supervised_baseline.py # Supervised baseline (cross-entropy)
│   ├── ts2vec_baseline.py     # Track D: TS2Vec (contrastive SSL)
│   ├── patchtst_baseline.py   # Track E: PatchTST (masked reconstruction)
│   └── evaluate.py            # Unified evaluation framework
└── results/                   # Generated at runtime (gitignored)
    ├── labels/                # HMM labels (.npy) for Mehul
    ├── results_table.csv
    ├── regime_plot.png
    ├── umap_comparison.png
    └── confusion_matrices.png
```

## For Mehul (FinJEPA Integration)

After running the pipeline, grab these files from `results/`:

```python
import numpy as np

# Load HMM labels
val_labels = np.load("results/labels/hmm_patch_labels_val.npy")
test_labels = np.load("results/labels/hmm_patch_labels_test.npy")

# Your FinJEPA representations should be (n_patches, 384)
# Then use the same linear probe evaluation:
from src.evaluate import evaluate_model
result = evaluate_model("FinJEPA", val_reprs, test_reprs, val_labels, test_labels)
```

## Key Design Decisions

- **3-state HMM** over 2-state: captures Bear/Sideways/Bull (richer than binary bull/bear)
- **20-day patches**: ~1 trading month, matches flowchart spec
- **384-dim representations**: all models output same dimensionality for fair comparison
- **6-layer transformer**: supervised and PatchTST match FinJEPA's architecture
- **Rolling z-score**: strictly causal (252-day lookback), prevents data leakage
