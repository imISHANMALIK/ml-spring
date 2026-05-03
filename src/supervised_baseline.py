"""
Supervised Baseline
====================
A transformer encoder trained end-to-end on HMM regime labels.
No pretraining, no self-supervision.

This is the "is SSL even necessary?" sanity check.
Uses the SAME architecture as FinJEPA's context encoder (6-layer transformer, 384-dim)
so the only difference is the training objective (cross-entropy vs JEPA loss).

Usage:
    from src.supervised_baseline import SupervisedBaseline, train_supervised
    model = SupervisedBaseline()
    train_supervised(model, train_loader, val_loader)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SupervisedBaseline(nn.Module):
    """6-layer transformer trained directly on regime classification.
    
    Same architecture as FinJEPA's context encoder, but trained with
    cross-entropy loss on HMM labels instead of JEPA loss.
    
    Architecture:
        - Patch embedding: Linear(patch_size → d_model)
        - Positional encoding: learnable
        - 6x TransformerEncoderLayer (d_model=384, nhead=6)
        - Global average pooling
        - Classification head: Linear(384 → n_classes)
    """
    
    def __init__(self, patch_size=20, d_model=384, n_heads=6, 
                 n_layers=6, n_classes=3, max_patches=64, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norm before pooling
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
    
    def get_representations(self, x):
        """Extract 384-dim representations (for evaluation).
        
        Args:
            x: (batch, n_patches, patch_size)
        Returns:
            (batch, d_model) — global average pooled representation
        """
        batch_size, n_patches, _ = x.shape
        
        # Embed patches
        x = self.patch_embed(x)  # (batch, n_patches, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :n_patches, :]
        
        # Transformer encoding
        x = self.encoder(x)  # (batch, n_patches, d_model)
        
        # Layer norm
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        return x
    
    def forward(self, x):
        """Forward pass for classification.
        
        Args:
            x: (batch, n_patches, patch_size)
        Returns:
            logits: (batch, n_classes)
        """
        x = self.get_representations(x)
        return self.classifier(x)


def train_supervised(model, train_dataset, val_dataset, 
                     n_epochs=100, lr=1e-4, batch_size=32,
                     weight_decay=1e-4, patience=15, device='auto'):
    """Train the supervised baseline.
    
    Args:
        model: SupervisedBaseline instance
        train_dataset: RegimeDataset for training
        val_dataset: RegimeDataset for validation
        n_epochs: max epochs
        lr: learning rate
        batch_size: batch size
        patience: early stopping patience
        device: 'auto', 'cuda', 'mps', or 'cpu'
        
    Returns:
        model: trained model (best checkpoint)
        history: dict with train_loss, val_loss, val_f1 per epoch
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    device = torch.device(device)
    model = model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Class weights for imbalanced regimes
    labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]
    
    n_classes = model.classifier[-1].out_features
    counts = np.bincount(labels, minlength=n_classes)
    weights = np.zeros(n_classes, dtype=np.float32)
    nonzero_mask = counts > 0
    weights[nonzero_mask] = 1.0 / counts[nonzero_mask]
    weights = weights / weights.sum() * nonzero_mask.sum()
    class_weights = torch.FloatTensor(weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    best_f1 = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        for patches, labels_batch in train_loader:
            patches, labels_batch = patches.to(device), labels_batch.to(device)
            
            logits = model(patches)
            loss = criterion(logits, labels_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for patches, labels_batch in val_loader:
                patches, labels_batch = patches.to(device), labels_batch.to(device)
                logits = model(patches)
                loss = criterion(logits, labels_batch)
                val_losses.append(loss.item())
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['val_f1'].append(val_f1)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train Loss: {np.mean(train_losses):.4f} | "
                  f"Val Loss: {np.mean(val_losses):.4f} | "
                  f"Val F1: {val_f1:.4f}")
        
        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (best F1: {best_f1:.4f})")
                break
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    
    print(f"\nBest validation macro-F1: {best_f1:.4f}")
    return model, history


def extract_representations(model, dataset, batch_size=64, device='auto'):
    """Extract frozen representations from the supervised model.
    
    Returns:
        representations: np.array of shape (n_samples, 384)
        labels: np.array of shape (n_samples,)
    """
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_reprs, all_labels = [], []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            reprs = model.get_representations(patches)
            all_reprs.append(reprs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_reprs), np.concatenate(all_labels)


if __name__ == "__main__":
    from data_pipeline import load_and_preprocess, RegimeDataset
    from hmm_labels import RegimeLabeler
    
    # Load data
    data = load_and_preprocess()
    
    # Get HMM labels
    labeler = RegimeLabeler(n_states=3)
    labels = labeler.fit_and_label(data)
    
    # Create datasets
    train_ds = RegimeDataset(data['splits']['train']['patches'], 
                             labels['patch_labels']['train'])
    val_ds = RegimeDataset(data['splits']['val']['patches'],
                           labels['patch_labels']['val'])
    
    # Train
    model = SupervisedBaseline()
    model, history = train_supervised(model, train_ds, val_ds)
    
    # Extract representations
    val_reprs, val_labels = extract_representations(model, val_ds)
    print(f"Val representations shape: {val_reprs.shape}")
