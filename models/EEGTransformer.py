import torch
import torch.nn as nn
import math

class EEGTransformer(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001, 
                 d_model=128, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        
        self.channels = channels
        self.samples = samples
        self.d_model = d_model
        
        # Patch embedding for EEG
        self.patch_size = 25  # 25 time points per patch
        self.num_patches = samples // self.patch_size
        
        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(channels * self.patch_size, d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, 1, channels, samples)
        batch_size = x.shape[0]
        x = x.squeeze(1)  # (batch, channels, samples)
        
        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size)  # (batch, channels, num_patches, patch_size)
        x = x.transpose(1, 2)  # (batch, num_patches, channels, patch_size)
        x = x.reshape(batch_size, self.num_patches, -1)  # (batch, num_patches, channels*patch_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, num_patches, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        return self.classifier(x)