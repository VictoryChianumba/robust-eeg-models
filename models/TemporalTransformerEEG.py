import torch
import torch.nn as nn
import math

class TTEEG(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        self.channels = channels
        self.d_model = d_model
        
        # Temporal embedding - treat each time point as a token
        self.temporal_embed = nn.Linear(channels, d_model)
        
        # Positional encoding for time steps
        self.pos_embed = nn.Parameter(torch.randn(1, samples, d_model))
        
        # Temporal transformer - processes time sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze(1)  # (batch, channels, samples)
        x = x.transpose(1, 2)  # (batch, samples, channels)
        
        # Temporal embedding
        x = self.temporal_embed(x)  # (batch, samples, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Temporal transformer
        x = self.temporal_transformer(x)  # (batch, samples, d_model)
        
        # Global temporal pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(x)