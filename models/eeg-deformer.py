
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

class EEGDeformer(nn.Module):
    """
    EEG-Deformer: Hierarchical Coarse-to-Fine Transformer
    
    Paper: "EEG-Deformer" (2024) - Dense convolutional transformer for BCIs
    
    Key Innovation: Hierarchical multi-scale attention with cross-scale information exchange
    - Coarse level: Global patterns, low resolution
    - Medium level: Regional patterns, medium resolution  
    - Fine level: Local patterns, full resolution
    """
    
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        samples: int = 1000,
        dropout_rate: float = 0.3,
        # Hierarchical parameters
        scales: List[int] = [8, 4, 1],  # Downsampling factors [coarse, medium, fine]
        d_model: int = 128,
        nhead: int = 8,
        num_layers_per_scale: int = 2,
        dim_feedforward: int = 512,
    ):
        super(EEGDeformer, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.samples = samples
        self.scales = scales
        self.d_model = d_model
        self.num_scales = len(scales)
        
        # =============== Initial Feature Extraction ===============
        
        # EEG-specific convolutional layers
        self.feature_extractor = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 32, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            
            # Spatial convolution
            nn.Conv2d(32, 64, (n_channels, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            
            # Additional temporal processing
            nn.Conv2d(64, d_model, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
        )
        
        # =============== Multi-Scale Attention Modules ===============
        
        self.scale_processors = nn.ModuleList()
        self.cross_scale_fusion = nn.ModuleList()
        
        for i, scale in enumerate(scales):
            # Multi-head attention for each scale
            scale_processor = MultiScaleAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers_per_scale,
                dim_feedforward=dim_feedforward,
                dropout=dropout_rate,
                scale_factor=scale
            )
            self.scale_processors.append(scale_processor)
            
            # Cross-scale fusion (except for the coarsest level)
            if i > 0:
                fusion_block = CrossScaleFusion(d_model)
                self.cross_scale_fusion.append(fusion_block)
        
        # =============== Feature Integration ===============
        
        self.feature_integration = nn.Sequential(
            nn.Conv1d(d_model, d_model // 2, 1),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # =============== Classification Head ===============
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model // 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical forward pass with coarse-to-fine processing
        
        Args:
            x: [batch_size, 1, n_channels, samples]
            
        Returns:
            logits: [batch_size, n_classes]
        """
        batch_size = x.shape[0]
        
        # =============== Initial Feature Extraction ===============
        features = self.feature_extractor(x)  # [B, d_model, 1, T]
        features = features.squeeze(2)  # [B, d_model, T]
        
        # =============== Multi-Scale Processing ===============
        
        scale_outputs = []
        
        # Process each scale independently
        for i, (scale, processor) in enumerate(zip(self.scales, self.scale_processors)):
            # Downsample features for current scale
            if scale > 1:
                downsampled = F.avg_pool1d(features, kernel_size=scale, stride=scale)
            else:
                downsampled = features
            
            # Process at current scale
            scale_output = processor(downsampled)
            scale_outputs.append(scale_output)
        
        # =============== Cross-Scale Information Exchange ===============
        
        # Start from coarsest and propagate to finer scales
        enhanced_outputs = [scale_outputs[0]]  # Coarse level (no enhancement needed)
        
        for i in range(1, len(scale_outputs)):
            # Get current scale output
            current_output = scale_outputs[i]
            
            # Get previous (coarser) scale output and upsample
            prev_output = enhanced_outputs[i-1]
            scale_ratio = self.scales[i-1] // self.scales[i]
            
            if scale_ratio > 1:
                # Upsample previous scale to current scale resolution
                upsampled_prev = F.interpolate(
                    prev_output, 
                    scale_factor=scale_ratio, 
                    mode='linear', 
                    align_corners=False
                )
            else:
                upsampled_prev = prev_output
            
            # Ensure sizes match (handle any rounding issues)
            if upsampled_prev.shape[-1] != current_output.shape[-1]:
                upsampled_prev = F.interpolate(
                    upsampled_prev,
                    size=current_output.shape[-1],
                    mode='linear',
                    align_corners=False
                )
            
            # Cross-scale fusion
            enhanced_output = self.cross_scale_fusion[i-1](current_output, upsampled_prev)
            enhanced_outputs.append(enhanced_output)
        
        # =============== Final Processing ===============
        
        # Use the finest scale output (full resolution with all hierarchical info)
        final_features = enhanced_outputs[-1]  # [B, d_model, T]
        
        # Feature integration
        integrated = self.feature_integration(final_features)  # [B, d_model//2, T]
        
        # Global pooling
        pooled = self.global_pool(integrated).squeeze(-1)  # [B, d_model//2]
        
        # Classification
        logits = self.classifier(pooled)  # [B, n_classes]
        
        return logits
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MultiScaleAttentionBlock(nn.Module):
    """Multi-head attention block for a specific scale"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        scale_factor: int
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Transformer layers for this scale
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Scale-specific normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features at specific scale
        
        Args:
            x: [batch_size, d_model, time_steps]
            
        Returns:
            Enhanced features at the same resolution
        """
        # Transpose for transformer: [B, T, d_model]
        x_transposed = x.transpose(1, 2)
        
        # Apply transformer
        enhanced = self.transformer(x_transposed)
        
        # Normalize
        enhanced = self.norm(enhanced)
        
        # Transpose back: [B, d_model, T]
        return enhanced.transpose(1, 2)


class CrossScaleFusion(nn.Module):
    """Fuse information from coarser scale to current scale"""
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Fusion mechanism
        self.fusion_conv = nn.Conv1d(d_model * 2, d_model, 1)
        self.gate = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, 1),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm1d(d_model)
        
    def forward(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """
        Fuse current scale with upsampled previous scale
        
        Args:
            current: Features at current scale [B, d_model, T]
            previous: Upsampled features from coarser scale [B, d_model, T]
            
        Returns:
            Fused features [B, d_model, T]
        """
        # Concatenate features
        concatenated = torch.cat([current, previous], dim=1)  # [B, 2*d_model, T]
        
        # Generate fusion weights
        gate_weights = self.gate(concatenated)  # [B, d_model, T]
        
        # Fuse features
        fused = self.fusion_conv(concatenated)  # [B, d_model, T]
        
        # Apply gating
        enhanced = current + gate_weights * fused
        
        # Normalize
        enhanced = self.norm(enhanced)
        
        return enhanced


# =============== Positional Encoding for EEG ===============

class EEGPositionalEncoding(nn.Module):
    """Positional encoding adapted for EEG temporal patterns"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# =============== Factory Function ===============

def create_eeg_deformer(
    n_channels: int = 22,
    n_classes: int = 4,
    samples: int = 1000,
    scales: Optional[List[int]] = None
) -> EEGDeformer:
    """
    Create EEG-Deformer model
    
    Args:
        n_channels: Number of EEG channels
        n_classes: Number of classes
        samples: Number of time samples
        scales: Downsampling factors [coarse, medium, fine]
        
    Returns:
        EEGDeformer model
    """
    if scales is None:
        scales = [8, 4, 1]  # Default hierarchy
        
    return EEGDeformer(
        n_channels=n_channels,
        n_classes=n_classes,
        samples=samples,
        scales=scales,
        d_model=128,
        nhead=8,
        num_layers_per_scale=2,
        dim_feedforward=512,
        dropout_rate=0.3
    )


if __name__ == "__main__":
    # Test the model
    model = create_eeg_deformer(n_channels=22, n_classes=4, samples=1000)
    
    # Test input
    x = torch.randn(16, 1, 22, 1000)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different scales
    print("\nTesting different hierarchical scales:")
    scales_configs = [
        [8, 4, 1],      # Default 3-level hierarchy
        [16, 8, 4, 1],  # 4-level hierarchy
        [4, 1],         # 2-level hierarchy
    ]
    
    for scales in scales_configs:
        model_test = create_eeg_deformer(scales=scales)
        out_test = model_test(x)
        print(f"Scales {scales} -> Parameters: {sum(p.numel() for p in model_test.parameters()):,}")
