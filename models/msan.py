
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

class MultiScaleAttentionNetwork(nn.Module):
    """
    Multi-Scale Attention Network for EEG Motor Imagery Classification
    
    Based on: "An end-to-end multi-task motor imagery EEG classification neural network 
    based on dynamic fusion of spectral-temporal features" (2024)
    
    Architecture:
    1. Compact CNN for spectral features
    2. GRU for temporal patterns
    3. Dynamic attention for channel fusion
    4. Multi-branch feature extraction
    """
    
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        samples: int = 1000,
        dropout_rate: float = 0.3,
        # CNN parameters
        n_filters_1: int = 32,
        n_filters_2: int = 64,
        n_filters_3: int = 128,
        # GRU parameters
        gru_hidden_size: int = 128,
        gru_num_layers: int = 2,
        # Attention parameters
        attention_dim: int = 64,
        num_heads: int = 8,
    ):
        super(MultiScaleAttentionNetwork, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.samples = samples
        
        # =============== Multi-Branch Feature Extraction ===============
        
        # Branch 1: Temporal features
        self.temporal_branch = TemporalFeatureBranch(
            n_channels=n_channels,
            n_filters=[n_filters_1, n_filters_2, n_filters_3],
            dropout_rate=dropout_rate
        )
        
        # Branch 2: Spectral features 
        self.spectral_branch = SpectralFeatureBranch(
            n_channels=n_channels,
            n_filters=[n_filters_1, n_filters_2, n_filters_3],
            dropout_rate=dropout_rate
        )
        
        # Branch 3: Spatial features
        self.spatial_branch = SpatialFeatureBranch(
            n_channels=n_channels,
            n_filters=[n_filters_1, n_filters_2, n_filters_3],
            dropout_rate=dropout_rate
        )
        
        # =============== Feature Fusion ===============
        
        # Calculate feature dimensions after CNN branches
        self.feature_dim = self._get_feature_dim()
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, n_filters_3),
            nn.BatchNorm1d(n_filters_3),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # =============== Temporal Sequence Modeling ===============
        
        self.gru = nn.GRU(
            input_size=n_filters_3,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=dropout_rate if gru_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # =============== Multi-Scale Attention Mechanism ===============
        
        self.attention_fusion = MultiScaleAttentionFusion(
            input_dim=gru_hidden_size * 2,  # Bidirectional GRU
            attention_dim=attention_dim,
            num_heads=num_heads,
            n_channels=n_channels,
            dropout_rate=dropout_rate
        )
        
        # =============== Classification Head ===============
        
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )
        
        self._initialize_weights()
    
    def _get_feature_dim(self) -> int:
        """Calculate feature dimension after CNN branches"""
        with torch.no_grad():
            x = torch.randn(1, 1, self.n_channels, self.samples)
            temp_out = self.temporal_branch(x)
            return temp_out.shape[1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, 1, n_channels, samples]
            
        Returns:
            logits: [batch_size, n_classes]
        """
        batch_size = x.shape[0]
        
        # =============== Multi-Branch Feature Extraction ===============
        
        temporal_features = self.temporal_branch(x)  # [B, feature_dim]
        spectral_features = self.spectral_branch(x)   # [B, feature_dim]
        spatial_features = self.spatial_branch(x)     # [B, feature_dim]
        
        # =============== Feature Fusion ===============
        
        # Concatenate branch features
        combined_features = torch.cat([
            temporal_features, 
            spectral_features, 
            spatial_features
        ], dim=1)  # [B, feature_dim * 3]
        
        # Fuse features
        fused_features = self.feature_fusion(combined_features)  # [B, n_filters_3]
        
        # =============== Temporal Sequence Processing ===============
        
        # Reshape for sequence processing (split into time steps)
        seq_len = 32  # Number of time steps for GRU processing
        if fused_features.shape[1] % seq_len != 0:
            # Pad to make divisible
            pad_size = seq_len - (fused_features.shape[1] % seq_len)
            fused_features = F.pad(fused_features, (0, pad_size))
        
        # Create sequence: [B, seq_len, feature_per_step]
        feature_per_step = fused_features.shape[1] // seq_len
        sequence = fused_features.view(batch_size, seq_len, feature_per_step)
        
        # GRU processing
        gru_output, _ = self.gru(sequence)  # [B, seq_len, gru_hidden_size * 2]
        
        # =============== Multi-Scale Attention ===============
        
        attended_features = self.attention_fusion(gru_output)  # [B, gru_hidden_size * 2]
        
        # =============== Classification ===============
        
        logits = self.classifier(attended_features)  # [B, n_classes]
        
        return logits
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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


class TemporalFeatureBranch(nn.Module):
    """Extract temporal features using 1D convolutions"""
    
    def __init__(self, n_channels: int, n_filters: List[int], dropout_rate: float):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # First temporal conv
            nn.Conv2d(1, n_filters[0], (1, 64), padding=(0, 32)),
            nn.BatchNorm2d(n_filters[0]),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            
            # Spatial conv
            nn.Conv2d(n_filters[0], n_filters[1], (n_channels, 1)),
            nn.BatchNorm2d(n_filters[1]),
            nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(dropout_rate),
            
            # Second temporal conv
            nn.Conv2d(n_filters[1], n_filters[2], (1, 32), padding=(0, 16)),
            nn.BatchNorm2d(n_filters[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)  # [B, n_filters[2], 1, 1]
        return x.view(x.size(0), -1)  # [B, n_filters[2]]


class SpectralFeatureBranch(nn.Module):
    """Extract spectral features using frequency-domain processing"""
    
    def __init__(self, n_channels: int, n_filters: List[int], dropout_rate: float):
        super().__init__()
        
        # Frequency bands of interest for motor imagery
        self.freq_filters = nn.ModuleList([
            nn.Conv2d(1, n_filters[0] // 3, (1, 16), padding=(0, 8)),  # Alpha (8-13 Hz)
            nn.Conv2d(1, n_filters[0] // 3, (1, 32), padding=(0, 16)), # Beta (13-30 Hz)
            nn.Conv2d(1, n_filters[0] // 3, (1, 64), padding=(0, 32)), # Gamma (30+ Hz)
        ])
        
        self.spatial_conv = nn.Conv2d(n_filters[0], n_filters[1], (n_channels, 1))
        self.batch_norm = nn.BatchNorm2d(n_filters[1])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(n_filters[1], n_filters[2], (1, 16), padding=(0, 8)),
            nn.BatchNorm2d(n_filters[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply frequency-specific filters
        freq_outputs = []
        for freq_filter in self.freq_filters:
            freq_out = F.relu(freq_filter(x))
            freq_outputs.append(freq_out)
        
        # Concatenate frequency bands
        x = torch.cat(freq_outputs, dim=1)  # [B, n_filters[0], n_channels, samples]
        
        # Spatial processing
        x = self.spatial_conv(x)  # [B, n_filters[1], 1, samples]
        x = F.relu(self.batch_norm(x))
        
        # Final processing
        x = self.final_conv(x)  # [B, n_filters[2], 1, 1]
        
        return x.view(x.size(0), -1)  # [B, n_filters[2]]


class SpatialFeatureBranch(nn.Module):
    """Extract spatial features focusing on channel relationships"""
    
    def __init__(self, n_channels: int, n_filters: List[int], dropout_rate: float):
        super().__init__()
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, n_filters[0], (n_channels, 1)),
            nn.BatchNorm2d(n_filters[0]),
            nn.ReLU(),
        )
        
        self.temporal_processing = nn.Sequential(
            nn.Conv2d(n_filters[0], n_filters[1], (1, 64), padding=(0, 32)),
            nn.BatchNorm2d(n_filters[1]),
            nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(n_filters[1], n_filters[2], (1, 32), padding=(0, 16)),
            nn.BatchNorm2d(n_filters[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial attention
        x = self.spatial_attention(x)  # [B, n_filters[0], 1, samples]
        
        # Temporal processing
        x = self.temporal_processing(x)  # [B, n_filters[2], 1, 1]
        
        return x.view(x.size(0), -1)  # [B, n_filters[2]]


class MultiScaleAttentionFusion(nn.Module):
    """
    Multi-scale attention mechanism for dynamic channel fusion
    """
    
    def __init__(
        self,
        input_dim: int,
        attention_dim: int,
        num_heads: int,
        n_channels: int,
        dropout_rate: float
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        
        # Multi-head attention components
        self.query = nn.Linear(input_dim, attention_dim, bias=False)
        self.key = nn.Linear(input_dim, attention_dim, bias=False)
        self.value = nn.Linear(input_dim, attention_dim, bias=False)
        
        # Channel attention
        self.channel_attention = ChannelAttentionModule(input_dim, n_channels)
        
        # Temporal attention
        self.temporal_attention = TemporalAttentionModule(input_dim, attention_dim)
        
        # Output projection
        self.out_proj = nn.Linear(attention_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Normalization
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, input_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # =============== Multi-Head Self-Attention ===============
        
        residual = x
        x = self.norm1(x)
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.attention_dim)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        x = residual + self.dropout(attn_output)
        
        # =============== Feed-Forward Network ===============
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        # =============== Channel and Temporal Attention ===============
        
        # Channel attention across sequence
        x_channel = self.channel_attention(x)  # [B, seq_len, input_dim]
        
        # Temporal attention
        x_temporal = self.temporal_attention(x_channel)  # [B, input_dim]
        
        return x_temporal


class ChannelAttentionModule(nn.Module):
    """Channel attention to focus on important feature channels"""
    
    def __init__(self, input_dim: int, n_channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(input_dim // reduction_ratio, input_dim, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, seq_len, input_dim]
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Global pooling across sequence dimension
        avg_pool = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1)  # [B, input_dim]
        max_pool = self.global_max_pool(x.transpose(1, 2)).squeeze(-1)  # [B, input_dim]
        
        # Generate attention weights
        avg_weights = self.fc(avg_pool)  # [B, input_dim]
        max_weights = self.fc(max_pool)  # [B, input_dim]
        
        # Combine and apply sigmoid
        channel_weights = self.sigmoid(avg_weights + max_weights)  # [B, input_dim]
        
        # Apply attention
        channel_weights = channel_weights.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, input_dim]
        
        return x * channel_weights


class TemporalAttentionModule(nn.Module):
    """Temporal attention to aggregate sequence information"""
    
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, input_dim]
        """
        # Compute attention weights for each time step
        attention_scores = self.attention_weights(x)  # [B, seq_len, 1]
        attention_weights = self.softmax(attention_scores)  # [B, seq_len, 1]
        
        # Weighted sum across time dimension
        attended_output = torch.sum(x * attention_weights, dim=1)  # [B, input_dim]
        
        return attended_output


# =============== Alternative Implementation: Spectral-Temporal CNN-GRU ===============

class SpectralTemporalCNNGRU(nn.Module):
    """
    Alternative implementation focusing on spectral-temporal feature extraction
    Based on the "dynamic fusion of spectral-temporal features" approach
    """
    
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        samples: int = 1000,
        dropout_rate: float = 0.3
    ):
        super().__init__()
        
        # =============== Compact CNN for Spectral Features ===============
        
        self.spectral_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            
            # Spatial conv
            nn.Conv2d(32, 64, (n_channels, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block  
            nn.Conv2d(64, 128, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate feature dimension
        self.feature_length = self._get_conv_output_length(samples)
        
        # =============== GRU for Temporal Patterns ===============
        
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # =============== Dynamic Channel Attention ===============
        
        self.channel_attention = nn.Sequential(
            nn.Linear(128, 64),  # 128 = 64 * 2 (bidirectional)
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        # =============== Classification ===============
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )
        
    def _get_conv_output_length(self, samples: int) -> int:
        """Calculate output length after convolutions"""
        # After first pooling (stride 4): samples // 4
        # After second pooling (stride 8): (samples // 4) // 8 = samples // 32
        return samples // 32
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 1, n_channels, samples]
        Returns:
            [batch_size, n_classes]
        """
        batch_size = x.shape[0]
        
        # =============== Spectral Feature Extraction ===============
        
        conv_features = self.spectral_cnn(x)  # [B, 128, 1, feature_length]
        conv_features = conv_features.squeeze(2)  # [B, 128, feature_length]
        
        # Transpose for GRU: [B, feature_length, 128]
        conv_features = conv_features.transpose(1, 2)
        
        # =============== Temporal Pattern Learning ===============
        
        gru_output, _ = self.gru(conv_features)  # [B, feature_length, 128]
        
        # =============== Dynamic Channel Attention ===============
        
        # Global average pooling across time
        pooled = torch.mean(gru_output, dim=1)  # [B, 128]
        
        # Generate channel attention weights
        attention_weights = self.channel_attention(pooled)  # [B, 128]
        
        # Apply attention to pooled features
        attended_features = pooled * attention_weights  # [B, 128]
        
        # =============== Classification ===============
        
        logits = self.classifier(attended_features)  # [B, n_classes]
        
        return logits


# =============== Factory Functions ===============

def create_multiscale_attention_network(
    n_channels: int = 22,
    n_classes: int = 4,
    samples: int = 1000,
    model_type: str = "full"
) -> nn.Module:
    """
    Create Multi-Scale Attention Network
    
    Args:
        n_channels: Number of EEG channels
        n_classes: Number of classes
        samples: Number of time samples
        model_type: "full" for complete model, "compact" for CNN-GRU version
        
    Returns:
        Multi-Scale Attention Network model
    """
    if model_type == "full":
        return MultiScaleAttentionNetwork(
            n_channels=n_channels,
            n_classes=n_classes,
            samples=samples,
            dropout_rate=0.3,
            n_filters_1=32,
            n_filters_2=64,
            n_filters_3=128,
            gru_hidden_size=128,
            gru_num_layers=2,
            attention_dim=64,
            num_heads=8
        )
    elif model_type == "compact":
        return SpectralTemporalCNNGRU(
            n_channels=n_channels,
            n_classes=n_classes,
            samples=samples,
            dropout_rate=0.3
        )
    else:
        raise ValueError("model_type must be 'full' or 'compact'")


if __name__ == "__main__":
    # Test both models
    print("Testing Multi-Scale Attention Network:")
    
    # Full model
    model_full = create_multiscale_attention_network(model_type="full")
    x = torch.randn(16, 1, 22, 1000)
    output_full = model_full(x)
    print(f"Full model - Input: {x.shape}, Output: {output_full.shape}")
    print(f"Full model parameters: {sum(p.numel() for p in model_full.parameters()):,}")
    
    # Compact model
    model_compact = create_multiscale_attention_network(model_type="compact")
    output_compact = model_compact(x)
    print(f"Compact model - Input: {x.shape}, Output: {output_compact.shape}")
    print(f"Compact model parameters: {sum(p.numel() for p in model_compact.parameters()):,}")
    
    # Test different input sizes
    print("\nTesting different input sizes:")
    for channels, samples in [(22, 1000), (64, 1125), (32, 750)]:
        model_test = create_multiscale_attention_network(
            n_channels=channels, 
            samples=samples, 
            model_type="compact"
        )
        x_test = torch.randn(8, 1, channels, samples)
        out_test = model_test(x_test)
        print(f"Channels: {channels}, Samples: {samples} -> Output: {out_test.shape}")
        print(f"Task: {task} -> Output shape: {out_test.shape}")
