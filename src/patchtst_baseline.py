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

Uses HuggingFace transformers if available, falls back to self-contained
implementation.

Usage:
    from src.patchtst_baseline import train_patchtst, extract_patchtst_representations
    model = train_patchtst(train_patches)
    representations = extract_patchtst_representations(model, test_patches)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import math
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Self-contained PatchTST Implementation
# ─────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """Embed patches into d_model dimensions with positional encoding."""
    
    def __init__(self, patch_size, d_model, max_patches=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, n_patches, patch_size)
        batch, n_patches, _ = x.shape
        x = self.proj(x)  # (batch, n_patches, d_model)
        x = x + self.pos_embed[:, :n_patches, :]
        x = self.norm(x)
        return self.dropout(x)


class PatchTSTEncoder(nn.Module):
    """Transformer encoder for PatchTST.
    
    Architecture matched to FinJEPA's context encoder:
    - 6 layers, 384-dim, 6 heads
    """
    
    def __init__(self, patch_size=20, d_model=384, n_heads=6, 
                 n_layers=6, max_patches=64, dropout=0.1, mask_ratio=0.4):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, d_model, max_patches, dropout)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Reconstruction head (predict raw patch values)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, patch_size)
        )
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def random_mask(self, x):
        """Apply random masking to patches.
        
        Args:
            x: (batch, n_patches, d_model) — embedded patches
            
        Returns:
            masked_x: (batch, n_patches, d_model) — with mask tokens
            mask: (batch, n_patches) — True = masked
        """
        batch, n_patches, d_model = x.shape
        n_mask = max(1, int(n_patches * self.mask_ratio))
        
        # Random mask indices per sample
        mask = torch.zeros(batch, n_patches, dtype=torch.bool, device=x.device)
        for i in range(batch):
            indices = torch.randperm(n_patches, device=x.device)[:n_mask]
            mask[i, indices] = True
        
        # Replace masked positions with mask token
        masked_x = x.clone()
        masked_x[mask] = self.mask_token.squeeze()
        
        return masked_x, mask
    
    def forward_pretrain(self, x):
        """Forward pass for pretraining (masked patch prediction).
        
        Args:
            x: (batch, n_patches, patch_size)
            
        Returns:
            loss: reconstruction MSE loss
            pred: (batch, n_patches, patch_size) predictions
            mask: (batch, n_patches) mask
        """
        # Embed patches
        embedded = self.patch_embed(x)  # (batch, n_patches, d_model)
        
        # Apply masking
        masked_embedded, mask = self.random_mask(embedded)
        
        # Encode (with mask tokens)
        encoded = self.encoder(masked_embedded)  # (batch, n_patches, d_model)
        encoded = self.norm(encoded)
        
        # Reconstruct masked patches
        pred = self.reconstruction_head(encoded)  # (batch, n_patches, patch_size)
        
        # Loss only on masked patches
        loss = F.mse_loss(pred[mask], x[mask])
        
        return loss, pred, mask
    
    def forward(self, x):
        """Forward pass for representation extraction (no masking).
        
        Args:
            x: (batch, n_patches, patch_size)
            
        Returns:
            representations: (batch, n_patches, d_model)
        """
        embedded = self.patch_embed(x)
        encoded = self.encoder(embedded)
        encoded = self.norm(encoded)
        return encoded
    
    def get_representations(self, x):
        """Extract pooled representation for classification.
        
        Args:
            x: (batch, n_patches, patch_size)
        Returns:
            (batch, d_model)
        """
        encoded = self.forward(x)
        return encoded.mean(dim=1)  # Global average pool


class PatchTSTModel:
    """PatchTST for time series representation learning."""
    
    def __init__(self, patch_size=20, d_model=384, n_heads=6,
                 n_layers=6, mask_ratio=0.4, device='auto'):
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = torch.device(device)
        self.patch_size = patch_size
        
        self.encoder = PatchTSTEncoder(
            patch_size=patch_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            mask_ratio=mask_ratio
        ).to(self.device)
    
    def fit(self, train_patches, n_epochs=200, lr=1e-4, batch_size=32, 
            context_len=12):
        """Pretrain PatchTST with masked patch reconstruction.
        
        Args:
            train_patches: np.array (n_patches, patch_size)
            context_len: number of patches per training window
        """
        # Create sliding windows of patches
        n_patches = len(train_patches)
        windows = []
        for i in range(n_patches - context_len + 1):
            window = train_patches[i:i+context_len]  # (context_len, patch_size)
            windows.append(window)
        
        train_data = np.array(windows)  # (n_windows, context_len, patch_size)
        print(f"PatchTST training data: {train_data.shape}")
        
        dataset = TensorDataset(torch.FloatTensor(train_data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        
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
        """Extract representations.
        
        Args:
            patches: np.array (n_patches, patch_size)
            
        Returns:
            representations: np.array (n_valid, d_model)
            where n_valid = n_patches - context_len + 1
        """
        self.encoder.eval()
        
        n_patches = len(patches)
        windows = []
        for i in range(n_patches - context_len + 1):
            windows.append(patches[i:i+context_len])
        
        data = np.array(windows)
        dataset = TensorDataset(torch.FloatTensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_reprs = []
        for (batch_x,) in loader:
            batch_x = batch_x.to(self.device)
            reprs = self.encoder.get_representations(batch_x)  # (batch, d_model)
            all_reprs.append(reprs.cpu().numpy())
        
        return np.concatenate(all_reprs, axis=0)
    
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
    """Train PatchTST on training patches.
    
    Args:
        train_patches: np.array (n_patches, patch_size=20)
        
    Returns:
        model: trained PatchTSTModel
    """
    model = PatchTSTModel(
        patch_size=train_patches.shape[1],
        d_model=d_model,
        n_heads=6,
        n_layers=6,
        mask_ratio=0.4,
        device=device
    )
    
    model.fit(train_patches, n_epochs=n_epochs, 
              batch_size=min(32, len(train_patches) // 2))
    return model


def extract_patchtst_representations(model, patches, context_len=12):
    """Extract patch-level representations from PatchTST.
    
    Returns:
        representations: np.array (n_valid_patches, d_model)
    """
    reprs = model.encode(patches, context_len=context_len)
    print(f"PatchTST representations: {reprs.shape}")
    return reprs


if __name__ == "__main__":
    from data_pipeline import load_and_preprocess
    
    data = load_and_preprocess()
    
    # Train PatchTST
    model = train_patchtst(data['splits']['train']['patches'], n_epochs=50)
    
    # Extract representations
    val_reprs = extract_patchtst_representations(model, data['splits']['val']['patches'])
    test_reprs = extract_patchtst_representations(model, data['splits']['test']['patches'])
    
    print(f"Val representations: {val_reprs.shape}")
    print(f"Test representations: {test_reprs.shape}")
