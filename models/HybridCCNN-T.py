import torch
import torch.nn as nn
import math

class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        
        # CNN feature extraction
        self.cnn_features = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 32, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            
            # Spatial convolution
            nn.Conv2d(32, 64, (channels, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            
            # Additional temporal features
            nn.Conv2d(64, d_model, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 2))
        )
        
        # Calculate sequence length after CNN
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            cnn_out = self.cnn_features(dummy)
            self.seq_len = cnn_out.shape[-1]
        
        # Positional encoding for transformer
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn_features(x)  # (batch, d_model, 1, seq_len)
        
        # Reshape for transformer
        batch_size = x.shape[0]
        x = x.squeeze(2).transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer processing
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(x)