import torch
import torch.nn as nn
import math

class EEGFormer(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        self.channels = channels
        self.samples = samples
        self.d_model = d_model
        
        # Channel embedding
        self.channel_embed = nn.Linear(samples, d_model)
        
        # Channel positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, channels, d_model))
        
        # Cross-channel transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.channel_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Temporal attention for each channel
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze(1)  # (batch, channels, samples)
        
        # Channel embedding
        x = self.channel_embed(x)  # (batch, channels, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Cross-channel transformer
        x = self.channel_transformer(x)  # (batch, channels, d_model)
        
        # Temporal attention
        attn_out, _ = self.temporal_attention(x, x, x)
        x = x + attn_out
        
        # Global pooling across channels
        x = x.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(x)