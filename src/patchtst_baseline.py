"""
PatchTST Baseline (Track E)
=============================
Masked patch prediction — reconstructs raw values of masked patches.
Unlike FinJEPA which predicts in representation space, PatchTST predicts
in input space (raw returns), forcing it to model noise.

Key differences from FinJEPA:
    - Predicts raw values (input space) vs latent representations
    - Random masking (bidirectional) vs causal prediction
    - Single encoder-decoder vs context encoder + predictor + EMA teacher
    - MSE loss in input space vs Smooth-L1 in latent space

The LayerWiseTransformerEncoder is imported from finjepa.py so both models
share the same extraction infrastructure — layer-to-layer comparisons are
apples-to-apples.

Usage:
    from src.patchtst_baseline import train_patchtst, extract_patchtst_representations
    model = train_patchtst(train_patches)
    representations = extract_patchtst_representations(model, test_patches)

    # Layer-wise extraction (new):
    layerwise = model.encode_layerwise(test_patches)  # list of 6 arrays
"""

from pyclbr import Class
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Shared layerwise encoder — defined in finjepa.py to keep one canonical source
from finjepa import LayerWiseTransformerEncoder


# ─────────────────────────────────────────────
# Patch embedding
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model, max_patches=64, dropout=0.1):
        super().__init__()
        self.proj      = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, n_patches, patch_size)
        _, n, _ = x.shape
        x = self.proj(x) + self.pos_embed[:, :n, :]
        return self.dropout(self.norm(x))


# ─────────────────────────────────────────────
# PatchTST Encoder — now with layerwise hooks
# ─────────────────────────────────────────────

class PatchTSTEncoder(nn.Module):
    """Transformer encoder for PatchTST.

    Architecture matched to FinJEPA's context encoder:
    - 6 layers, 384-dim, 6 heads

    The internal nn.TransformerEncoder has been replaced by
    LayerWiseTransformerEncoder so we can extract hidden states at
    every layer depth — the same hook used by FinJEPA — ensuring the
    layer-wise probing comparison is structurally identical.
    """

    def __init__(self, patch_size=20, d_model=384, n_heads=6,
                 n_layers=6, max_patches=64, dropout=0.1, mask_ratio=0.4):
        super().__init__()

        self.d_model    = d_model
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.n_layers   = n_layers

        self.patch_embed = PatchEmbedding(patch_size, d_model, max_patches, dropout)

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ← previously nn.TransformerEncoder; now exposes per-layer outputs
        self.encoder = LayerWiseTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )

        # Reconstruction head: predicts raw patch values (input space)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, patch_size)
        )

        self.norm = nn.LayerNorm(d_model)

    # ── Masking ──────────────────────────────────────────────────────────────

    def random_mask(self, x):
        """Apply random masking to embedded patches.

        Returns:
            masked_x: (batch, n_patches, d_model) — mask tokens inserted
            mask:     (batch, n_patches) bool — True = masked position
        """
        batch, n_patches, _ = x.shape
        n_mask = max(1, int(n_patches * self.mask_ratio))

        mask = torch.zeros(batch, n_patches, dtype=torch.bool, device=x.device)
        for i in range(batch):
            idx = torch.randperm(n_patches, device=x.device)[:n_mask]
            mask[i, idx] = True

        masked_x = x.clone()
        masked_x[mask] = self.mask_token.squeeze()
        return masked_x, mask

    # ── Pre-training forward ──────────────────────────────────────────────────

    def forward_pretrain(self, x):
        """Masked patch reconstruction — PatchTST's training objective.

        Predicts raw return values for masked positions. This forces the
        encoder to retain noise information (the antithesis of FinJEPA).

        Returns: (loss, pred, mask)
        """
        embedded = self.patch_embed(x)
        masked_embedded, mask = self.random_mask(embedded)

        # Standard (non-layerwise) forward for pretraining
        encoded = self.encoder(masked_embedded)
        encoded = self.norm(encoded)

        pred = self.reconstruction_head(encoded)
        loss = F.mse_loss(pred[mask], x[mask])
        return loss, pred, mask

    # ── Representation forward (no masking) ──────────────────────────────────

    def forward(self, x):
        """Clean forward pass — no masking, returns (B, n_patches, d_model)."""
        embedded = self.patch_embed(x)
        encoded  = self.encoder(embedded)
        return self.norm(encoded)

    def get_representations(self, x):
        """Global-average-pooled final representation. Returns (B, d_model)."""
        return self.forward(x).mean(dim=1)

    # ── Layerwise extraction ──────────────────────────────────────────────────

    def forward_layerwise(self, x):
        """Extract hidden state after every transformer layer.

        Mirrors FinJEPAEncoder.forward_layerwise exactly:
          - Post-layer-norm is applied only to the final layer's output.
          - Intermediate layers return raw post-residual activations.

        Returns: list of (B, n_patches, d_model), one tensor per layer.
        """
        embedded = self.patch_embed(x)
        hidden_states = self.encoder.forward_layerwise(embedded)
        # Match FinJEPA's convention: final norm only on last layer
        hidden_states[-1] = self.norm(hidden_states[-1])
        return hidden_states


# ─────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────

class PatchTSTModel:
    """PatchTST for time series representation learning."""

    def __init__(self, patch_size=20, d_model=384, n_heads=6,
                 n_layers=6, mask_ratio=0.4, device='auto'):
        if device == 'auto':
            if torch.cuda.is_available():           device = 'cuda'
            elif torch.backends.mps.is_available(): device = 'mps'
            else:                                   device = 'cpu'

        self.device     = torch.device(device)
        self.patch_size = patch_size
        self.n_layers   = n_layers

        self.encoder = PatchTSTEncoder(
            patch_size=patch_size, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, mask_ratio=mask_ratio
        ).to(self.device)

    def fit(self, train_patches, n_epochs=200, lr=1e-4,
            batch_size=32, context_len=12):
        """Pretrain PatchTST with masked patch reconstruction."""
        n = len(train_patches)
        windows = np.array([train_patches[i:i+context_len]
                            for i in range(n - context_len + 1)])
        print(f"PatchTST training data: {windows.shape}")

        loader = DataLoader(TensorDataset(torch.FloatTensor(windows)),
                            batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        self.encoder.train()
        for epoch in range(n_epochs):
            losses = []
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                loss, _, _ = self.encoder.forward_pretrain(batch_x)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            scheduler.step()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"PatchTST Epoch {epoch+1:3d}/{n_epochs} | "
                      f"Recon Loss: {np.mean(losses):.6f}")

        print("PatchTST pretraining complete.")

    @torch.no_grad()
    def encode(self, patches, context_len=12, batch_size=64):
        """Final-layer pooled representations. Returns (n_windows, d_model)."""
        self.encoder.eval()
        n = len(patches)
        windows = np.array([patches[i:i+context_len]
                            for i in range(n - context_len + 1)])
        loader = DataLoader(TensorDataset(torch.FloatTensor(windows)),
                            batch_size=batch_size, shuffle=False)

        reprs = []
        for (x,) in loader:
            reprs.append(self.encoder.get_representations(x.to(self.device))
                         .cpu().numpy())
        return np.concatenate(reprs, axis=0)

    @torch.no_grad()
    def encode_layerwise(self, patches, context_len=12, batch_size=64):
        """Extract per-layer global-average-pooled representations.

        Returns: list of n_layers np.arrays, each (n_windows, d_model).
        Layer index 0 → Layer 1, ..., index 5 → Layer 6.

        Uses the same global-average-pool convention as FinJEPAModel so
        probe results are directly comparable.
        """
        self.encoder.eval()
        n = len(patches)
        windows = np.array([patches[i:i+context_len]
                            for i in range(n - context_len + 1)])
        loader = DataLoader(TensorDataset(torch.FloatTensor(windows)),
                            batch_size=batch_size, shuffle=False)

        layer_buckets = [[] for _ in range(self.n_layers)]

        for (x,) in loader:
            x = x.to(self.device)
            # hidden_states[i]: (B, context_len, d_model) at layer i+1
            hidden_states = self.encoder.forward_layerwise(x)
            for i, h in enumerate(hidden_states):
                layer_buckets[i].append(h.mean(dim=1).cpu().numpy())

        return [np.concatenate(b, axis=0) for b in layer_buckets]

    def save(self, path):
        torch.save(self.encoder.state_dict(), path)
        print(f"Saved PatchTST to {path}")

    def load(self, path):
        self.encoder.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded PatchTST from {path}")


# ─────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────

def train_patchtst(train_patches, d_model=384, n_epochs=200, device='auto'):
    model = PatchTSTModel(
        patch_size=train_patches.shape[1],
        d_model=d_model, n_heads=6, n_layers=6,
        mask_ratio=0.4, device=device
    )
    model.fit(train_patches, n_epochs=n_epochs,
              batch_size=min(32, len(train_patches) // 2))
    return model


def extract_patchtst_representations(model, patches, context_len=12):
    """Legacy API — final-layer pooled representations."""
    reprs = model.encode(patches, context_len=context_len)
    print(f"PatchTST representations: {reprs.shape}")
    return reprs


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__))
    from data_pipeline import load_and_preprocess

    data  = load_and_preprocess()
    model = train_patchtst(data['splits']['train']['patches'], n_epochs=50)

    val_reprs  = extract_patchtst_representations(model, data['splits']['val']['patches'])
    test_reprs = extract_patchtst_representations(model, data['splits']['test']['patches'])

    val_layerwise  = model.encode_layerwise(data['splits']['val']['patches'])
    print(f"Layerwise val shapes: {[a.shape for a in val_layerwise]}")
