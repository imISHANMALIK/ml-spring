"""
FinJEPA — Joint Embedding Predictive Architecture for Market Regime Detection
==============================================================================
Architecture:
    - Context Encoder (f_θ):  6-layer Transformer, online (trained by optimizer)
    - Target Encoder (f_ξ):   EMA copy of context encoder, updated adaptively based
                               on batch volatility — high-vol markets get lower τ so
                               the target adapts faster instead of staying anchored
                               to stale, calm-market representations.
    - Predictor (g_φ):        Cross-attention module that takes context encoder output
                               and predicts target-patch representations.

Training objective: Smooth-L1 loss between g_φ(f_θ(context)) and stop_grad(f_ξ(target))

The JEPA setup is critical: by predicting in representation space (not raw input space),
the model is forced to filter noise — if a representation encodes noise, predicting it
requires overfitting. Abstract, regime-level structure is predictable; tick-level noise is not.

Exports:
    LayerWiseTransformerEncoder — shared with patchtst_baseline
    FinJEPAModel                — full model
    train_finjepa               — high-level API
    extract_finjepa_representations  — legacy single-vector API
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Shared: LayerWiseTransformerEncoder
# Imported by patchtst_baseline.py — define once, use in both models.
# ─────────────────────────────────────────────────────────────────────────────

class LayerWiseTransformerEncoder(nn.Module):
    """Drop-in replacement for nn.TransformerEncoder that exposes per-layer hidden states.

    Standard forward() is identical to nn.TransformerEncoder. The extra method
    forward_layerwise() returns a list of tensors — one per layer — so we can
    probe what information is linearly decodable at each depth.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,   # pre-norm (more stable for SSL)
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward — returns only the final layer output."""
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_layerwise(self, x: torch.Tensor) -> list:
        """Returns list of hidden states after every layer (layers 1 … n_layers).

        Each entry is a clone so in-place ops on x don't corrupt earlier captures.
        Shape per entry: (batch, n_patches, d_model).
        """
        hidden_states = []
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x.clone())
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Patch embedding — shared by context and target encoders
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int,
                 max_patches: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj     = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_patches, patch_size)
        _, n, _ = x.shape
        x = self.proj(x) + self.pos_embed[:, :n, :]
        return self.dropout(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Context / Target Encoder (shared architecture, different update rules)
# ─────────────────────────────────────────────────────────────────────────────

class FinJEPAEncoder(nn.Module):
    """Patch-embedding → 6-layer pre-norm Transformer → LayerNorm.

    The final LayerNorm is applied only to the last layer in forward_layerwise(),
    so all six probing surfaces are consistent (no intermediate norm artifacts).
    """

    def __init__(self, patch_size: int, d_model: int, n_heads: int,
                 n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, d_model, dropout=dropout)
        self.encoder     = LayerWiseTransformerEncoder(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            dim_feedforward=4 * d_model, dropout=dropout
        )
        self.norm        = nn.LayerNorm(d_model)
        self.n_layers    = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_patches, patch_size)  →  (B, n_patches, d_model)"""
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.norm(x)

    def forward_layerwise(self, x: torch.Tensor) -> list:
        """Returns one (B, n_patches, d_model) tensor per layer.

        The final norm is applied only to the last entry so the last layer's
        probing surface matches what the JEPA loss actually optimizes.
        """
        x = self.patch_embed(x)
        hidden_states = self.encoder.forward_layerwise(x)
        # Apply the post-encoder norm to the final layer only
        hidden_states[-1] = self.norm(hidden_states[-1])
        return hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Predictor — cross-attention from context to target positions
# ─────────────────────────────────────────────────────────────────────────────

class FinJEPAPredictor(nn.Module):
    """Predicts target-patch representations from context encoder output.

    Learnable position queries (one per target slot) attend to the context
    representations via cross-attention. This forces the predictor to learn
    *which* abstract features from the context are predictive for future
    positions — not low-level return values.
    """

    def __init__(self, d_model: int, n_heads: int,
                 n_target_patches: int, dropout: float = 0.1):
        super().__init__()
        # One learnable query per target-patch slot
        self.target_queries = nn.Parameter(
            torch.randn(1, n_target_patches, d_model) * 0.02
        )
        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, context_repr: torch.Tensor) -> torch.Tensor:
        """context_repr: (B, context_len, d_model)  →  (B, n_target, d_model)"""
        B = context_repr.shape[0]
        q  = self.target_queries.expand(B, -1, -1)
        kv = self.norm_kv(context_repr)

        # Cross-attention: target queries attend to context
        attn_out, _ = self.cross_attn(self.norm_q(q), kv, kv)
        q = q + attn_out                     # residual
        q = q + self.ffn(self.norm_ffn(q))   # FFN + residual
        return self.out_norm(q)


# ─────────────────────────────────────────────────────────────────────────────
# Full FinJEPA Model
# ─────────────────────────────────────────────────────────────────────────────

class FinJEPAModel:
    """FinJEPA: context encoder + predictor + adaptive-EMA target encoder.

    EMA schedule:
        τ(vol) = τ_max − (τ_max − τ_min) × tanh(λ × vol)

    Intuition: in calm markets (low vol) we want a stable target (high τ).
    In volatile markets the target should adapt faster (lower τ) so it doesn't
    anchor the predictor to a stale, pre-crash regime.
    """

    def __init__(self, patch_size: int = 20, d_model: int = 384,
                 n_heads: int = 6, n_layers: int = 6,
                 context_len: int = 12, target_len: int = 4,
                 dropout: float = 0.1,
                 tau_min: float = 0.990, tau_max: float = 0.999,
                 vol_sensitivity: float = 1.5,
                 device: str = 'auto'):

        if device == 'auto':
            if torch.cuda.is_available():      device = 'cuda'
            elif torch.backends.mps.is_available(): device = 'mps'
            else:                              device = 'cpu'

        self.device      = torch.device(device)
        self.context_len = context_len
        self.target_len  = target_len
        self.n_layers    = n_layers
        self.tau_min     = tau_min
        self.tau_max     = tau_max
        self.vol_sensitivity = vol_sensitivity
        self.current_tau = tau_max

        # --- Build sub-modules ---
        self.context_encoder = FinJEPAEncoder(
            patch_size, d_model, n_heads, n_layers, dropout
        ).to(self.device)

        # Target encoder: EMA copy, never updated by optimizer
        self.target_encoder = FinJEPAEncoder(
            patch_size, d_model, n_heads, n_layers, dropout
        ).to(self.device)
        self._sync_target()
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        self.predictor = FinJEPAPredictor(
            d_model, n_heads, target_len, dropout
        ).to(self.device)

    # ── EMA ──────────────────────────────────────────────────────────────────

    def _sync_target(self):
        """Hard-copy context → target (used at initialization)."""
        for p_c, p_t in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.copy_(p_c.data)

    @torch.no_grad()
    def _ema_update(self, batch_vol: float) -> float:
        """Adaptive EMA step. Returns the τ used this step."""
        tau = self.tau_max - (self.tau_max - self.tau_min) * math.tanh(
            self.vol_sensitivity * batch_vol
        )
        tau = float(np.clip(tau, self.tau_min, self.tau_max))
        self.current_tau = tau

        for p_c, p_t in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_c.data, alpha=1.0 - tau)
        return tau

    # ── Forward ──────────────────────────────────────────────────────────────

    def _forward_step(self, context: torch.Tensor,
                      target: torch.Tensor):
        """Single forward pass; returns (loss, batch_volatility)."""
        # Batch volatility from the raw z-scored returns in the context
        batch_vol = float(context.std().item())

        # Encode context (online encoder)
        ctx_repr = self.context_encoder(context)          # (B, ctx_len, d)

        # Encode target (EMA encoder, stop gradient)
        with torch.no_grad():
            tgt_repr = self.target_encoder(target)        # (B, tgt_len, d)

        # Predict target representations
        pred_repr = self.predictor(ctx_repr)              # (B, tgt_len, d)

        # JEPA loss: Smooth-L1 in representation space
        pred_loss = F.smooth_l1_loss(pred_repr, tgt_repr.detach())

        # Variance regularization: prevents representation collapse
        # (no negative samples in JEPA, so we explicitly penalise low std)
        var_reg = torch.relu(1.0 - ctx_repr.std(dim=0)).mean()

        loss = pred_loss + 0.05 * var_reg
        return loss, batch_vol

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, train_patches: np.ndarray, n_epochs: int = 200,
            lr: float = 1e-4, batch_size: int = 32) -> 'FinJEPAModel':
        """Pretrain FinJEPA with the JEPA objective.

        Args:
            train_patches: (n_patches, patch_size) normalized returns
        """
        total_len = self.context_len + self.target_len
        n = len(train_patches)

        contexts = np.array([train_patches[i:i + self.context_len]
                             for i in range(n - total_len + 1)])
        targets  = np.array([train_patches[i + self.context_len:i + total_len]
                             for i in range(n - total_len + 1)])

        print(f"FinJEPA training data: {contexts.shape} context, {targets.shape} target")

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(contexts), torch.FloatTensor(targets)),
            batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Only context encoder + predictor are trainable; target is EMA-only
        optimizer = torch.optim.AdamW(
            list(self.context_encoder.parameters()) +
            list(self.predictor.parameters()),
            lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )

        for epoch in range(n_epochs):
            self.context_encoder.train()
            self.predictor.train()
            losses, taus = [], []

            for ctx_batch, tgt_batch in loader:
                ctx_batch = ctx_batch.to(self.device)
                tgt_batch = tgt_batch.to(self.device)

                loss, batch_vol = self._forward_step(ctx_batch, tgt_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.context_encoder.parameters()) +
                    list(self.predictor.parameters()), 1.0
                )
                optimizer.step()

                tau = self._ema_update(batch_vol)   # EMA after optimizer step
                losses.append(loss.item())
                taus.append(tau)

            scheduler.step()
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"FinJEPA Epoch {epoch+1:3d}/{n_epochs} | "
                      f"JEPA Loss: {np.mean(losses):.6f} | "
                      f"τ (mean): {np.mean(taus):.5f}")

        print("FinJEPA pretraining complete.")
        return self

    # ── Inference helpers ────────────────────────────────────────────────────

    def _make_windows(self, patches: np.ndarray) -> np.ndarray:
        n = len(patches)
        return np.array([patches[i:i + self.context_len]
                         for i in range(n - self.context_len + 1)])

    @torch.no_grad()
    def encode(self, patches: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Final-layer global-average-pooled representations.

        Returns: (n_windows, d_model)
        """
        self.context_encoder.eval()
        windows = self._make_windows(patches)
        loader  = DataLoader(TensorDataset(torch.FloatTensor(windows)),
                             batch_size=batch_size, shuffle=False)
        reprs = []
        for (x,) in loader:
            out = self.context_encoder(x.to(self.device)).mean(dim=1)
            reprs.append(out.cpu().numpy())
        return np.concatenate(reprs, axis=0)

    @torch.no_grad()
    def encode_layerwise(self, patches: np.ndarray,
                         batch_size: int = 64) -> list:
        """Extract per-layer global-average-pooled representations.

        Returns: list of n_layers np.arrays, each (n_windows, d_model).
        Layer index 0 → Layer 1, ..., index 5 → Layer 6.
        """
        self.context_encoder.eval()
        windows = self._make_windows(patches)
        loader  = DataLoader(TensorDataset(torch.FloatTensor(windows)),
                             batch_size=batch_size, shuffle=False)

        # Accumulate per-layer outputs across batches
        layer_buckets = [[] for _ in range(self.n_layers)]

        for (x,) in loader:
            x = x.to(self.device)
            # hidden_states[i]: (B, context_len, d_model) at layer i+1
            hidden_states = self.context_encoder.forward_layerwise(x)
            for i, h in enumerate(hidden_states):
                # Global average pool: (B, d_model)
                layer_buckets[i].append(h.mean(dim=1).cpu().numpy())

        return [np.concatenate(b, axis=0) for b in layer_buckets]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({
            'context_encoder': self.context_encoder.state_dict(),
            'target_encoder':  self.target_encoder.state_dict(),
            'predictor':       self.predictor.state_dict(),
        }, path)
        print(f"Saved FinJEPA to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.context_encoder.load_state_dict(ckpt['context_encoder'])
        self.target_encoder.load_state_dict(ckpt['target_encoder'])
        self.predictor.load_state_dict(ckpt['predictor'])
        print(f"Loaded FinJEPA from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# High-level API
# ─────────────────────────────────────────────────────────────────────────────

def train_finjepa(train_patches: np.ndarray, d_model: int = 384,
                  n_epochs: int = 200, device: str = 'auto') -> FinJEPAModel:
    model = FinJEPAModel(
        patch_size=train_patches.shape[1],
        d_model=d_model, n_heads=6, n_layers=6,
        context_len=12, target_len=4,
        device=device
    )
    model.fit(train_patches, n_epochs=n_epochs,
              batch_size=min(32, len(train_patches) // 2))
    return model


def extract_finjepa_representations(model: FinJEPAModel,
                                    patches: np.ndarray) -> np.ndarray:
    """Legacy single-vector API — returns final-layer pooled representations."""
    reprs = model.encode(patches)
    print(f"FinJEPA representations: {reprs.shape}")
    return reprs
