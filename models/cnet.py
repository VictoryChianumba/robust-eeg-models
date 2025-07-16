
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class CTNet(nn.Module):
    """
    Convolutional Transformer Network for EEG Motor Imagery Classification
    
    Paper: "CTNet: a convolutional transformer network for EEG-based motor imagery classification"
    Scientific Reports 2024
    
    Architecture:
    1. EEGNet-style Convolutional Module
    2. Transformer Encoder with Multi-Head Attention
    3. Classification Head
    """
    
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        samples: int = 1000,
        dropout_rate: float = 0.5,
        # Convolutional parameters
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        # Transformer parameters
        d_model: int = 128,
        nhead: int = 2,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        token_size: int = 32,
    ):
        super(CTNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.samples = samples
        self.d_model = d_model
        self.token_size = token_size
        
        # =============== Convolutional Module (EEGNet-style) ===============
        
        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Block 2: Spatial Convolution (Depthwise)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 3: Separable Convolution
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate feature dimensions after convolutions
        self.feature_length = self._get_conv_output_size()
        
        # =============== Transformer Module ===============
        
        # Patch embedding projection
        self.patch_embed = nn.Linear(self.token_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=5000)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # =============== Classification Head ===============
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )
        
        self._initialize_weights()
    
    def _get_conv_output_size(self) -> int:
        """Calculate the output size after convolutional layers"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.samples)
            x = self._conv_forward(x)
            return x.shape[-1]
    
    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers"""
        # Block 1: Temporal Convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2: Spatial Convolution
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 3: Separable Convolution
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        return x
    
    def _create_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Convert feature maps to tokens for transformer"""
        batch_size, channels, height, length = x.shape
        
        # Flatten spatial dimensions
        x = x.view(batch_size, channels * height, length)  # [B, C*H, L]
        
        # Create tokens by reshaping
        n_tokens = length // self.token_size
        if length % self.token_size != 0:
            # Pad to make divisible by token_size
            pad_size = self.token_size - (length % self.token_size)
            x = F.pad(x, (0, pad_size))
            n_tokens = (length + pad_size) // self.token_size
        
        # Reshape to tokens: [B, C*H, n_tokens, token_size]
        x = x.view(batch_size, channels * height, n_tokens, self.token_size)
        
        # Average across channels: [B, n_tokens, token_size]
        x = x.mean(dim=1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, 1, n_channels, samples]
            
        Returns:
            logits: Output tensor of shape [batch_size, n_classes]
        """
        # =============== Convolutional Feature Extraction ===============
        conv_features = self._conv_forward(x)  # [B, F2, 1, length]
        
        # =============== Token Creation ===============
        tokens = self._create_tokens(conv_features)  # [B, n_tokens, token_size]
        
        # =============== Transformer Processing ===============
        # Project to transformer dimension
        tokens = self.patch_embed(tokens)  # [B, n_tokens, d_model]
        
        # Add positional encoding
        tokens = self.pos_encoding(tokens)
        
        # Transformer encoding
        transformer_out = self.transformer(tokens)  # [B, n_tokens, d_model]
        
        # =============== Global Pooling and Classification ===============
        # Global average pooling across tokens
        pooled = self.global_avg_pool(transformer_out.transpose(1, 2)).squeeze(-1)  # [B, d_model]
        
        # Classification
        logits = self.classifier(pooled)  # [B, n_classes]
        
        return logits
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# =============== Usage Example ===============

def create_ctnet(n_channels: int = 22, n_classes: int = 4, samples: int = 1000) -> CTNet:
    """
    Create CTNet model with default parameters
    
    Args:
        n_channels: Number of EEG channels
        n_classes: Number of classes (e.g., 4 for BCI IV-2a)
        samples: Number of time samples
        
    Returns:
        CTNet model
    """
    return CTNet(
        n_channels=n_channels,
        n_classes=n_classes,
        samples=samples,
        dropout_rate=0.5,
        F1=8,
        D=2,
        F2=16,
        d_model=128,
        nhead=2,
        num_layers=6,
        dim_feedforward=256,
        token_size=32
    )


if __name__ == "__main__":
    # Test the model
    model = create_ctnet(n_channels=22, n_classes=4, samples=1000)
    
    # Test input
    x = torch.randn(32, 1, 22, 1000)  # [batch_size, 1, channels, samples]
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different input sizes
    print("\nTesting different input sizes:")
    for channels, samples in [(22, 1000), (64, 1125), (32, 750)]:
        model_test = create_ctnet(n_channels=channels, n_classes=4, samples=samples)
        x_test = torch.randn(16, 1, channels, samples)
        out_test = model_test(x_test)
        print(f"Channels: {channels}, Samples: {samples} -> Output: {out_test.shape}")
