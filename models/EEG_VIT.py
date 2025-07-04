import torch
import torch.nn as nn
import math

class EEGViT(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001,
                 patch_size=50, d_model=256, nhead=8, num_layers=12, dropout=0.1):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = samples // patch_size
        self.d_model = d_model
        
        # Channel embedding (treat each channel as a "color" channel)
        self.channel_embed = nn.Linear(channels, d_model // 2)
        
        # Temporal embedding  
        self.temporal_embed = nn.Linear(patch_size, d_model // 2)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze(1)  # (batch, channels, samples)
        
        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size)  # (batch, channels, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3)  # (batch, num_patches, channels, patch_size)
        
        # Embed channels and time separately then concatenate
        channel_emb = self.channel_embed(x.mean(dim=3))  # (batch, num_patches, d_model//2)
        temporal_emb = self.temporal_embed(x.mean(dim=2))  # (batch, num_patches, d_model//2)
        
        # Combine embeddings
        x = torch.cat([channel_emb, temporal_emb], dim=-1)  # (batch, num_patches, d_model)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, d_model)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification (use class token)
        return self.head(x[:, 0])