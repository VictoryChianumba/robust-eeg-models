
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class MambaConfig:
    """Configuration for EEGMamba model"""
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: str = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    bias: bool = False
    conv_bias: bool = True
    

class EEGMamba(nn.Module):
    """
    EEGMamba: Bidirectional State Space Model with Mixture of Experts for EEG Classification
    
    Paper: "EEGMamba: Bidirectional State Space Model with Mixture of Experts for EEG Multi-task Classification" (2024)
    
    Key Components:
    1. ST-Adaptive Module: Handles variable signal lengths and channel counts
    2. Bidirectional Mamba: Efficient long-sequence modeling O(n) complexity
    3. Task-aware MoE: Specialized experts for different EEG tasks
    4. Universal Expert: Captures commonalities across tasks
    """
    
    def __init__(
        self,
        n_channels: int = 22,
        n_classes: int = 4,
        samples: int = 1000,
        config: Optional[MambaConfig] = None,
        num_experts: int = 4,
        num_mamba_layers: int = 6,
        dropout_rate: float = 0.1,
        task_type: str = "motor_imagery",  # motor_imagery, emotion, seizure, sleep
    ):
        super(EEGMamba, self).__init__()
        
        if config is None:
            config = MambaConfig()
        
        self.config = config
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.samples = samples
        self.task_type = task_type
        self.num_experts = num_experts
        
        # =============== ST-Adaptive Module ===============
        self.st_adaptive = STAdaptiveModule(
            n_channels=n_channels,
            d_model=config.d_model,
            samples=samples
        )
        
        # =============== Bidirectional Mamba Blocks ===============
        self.mamba_layers = nn.ModuleList([
            BidirectionalMambaBlock(config) for _ in range(num_mamba_layers)
        ])
        
        # =============== Task-aware Mixture of Experts ===============
        self.moe = TaskAwareMoE(
            d_model=config.d_model,
            num_experts=num_experts,
            num_classes=n_classes,
            task_type=task_type,
            dropout=dropout_rate
        )
        
        # =============== Classification Head ===============
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(config.d_model, n_classes)
        
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through EEGMamba
        
        Args:
            x: Input tensor [batch_size, 1, n_channels, samples]
            task_id: Optional task identifier for MoE routing
            
        Returns:
            logits: [batch_size, n_classes]
        """
        # =============== ST-Adaptive Processing ===============
        x = self.st_adaptive(x)  # [B, seq_len, d_model]
        
        # =============== Bidirectional Mamba Processing ===============
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)  # [B, seq_len, d_model]
        
        # =============== Task-aware MoE ===============
        x = self.moe(x, task_id)  # [B, seq_len, d_model]
        
        # =============== Global Pooling and Classification ===============
        x = self.norm(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # [B, d_model]
        
        x = self.dropout(x)
        logits = self.classifier(x)  # [B, n_classes]
        
        return logits
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class STAdaptiveModule(nn.Module):
    """
    Spatio-Temporal-Adaptive Module
    Handles variable signal lengths and channel counts through spatial-adaptive convolution
    """
    
    def __init__(self, n_channels: int, d_model: int, samples: int):
        super().__init__()
        
        self.n_channels = n_channels
        self.d_model = d_model
        
        # Spatial-adaptive convolution
        self.spatial_conv = nn.Conv2d(1, 32, (n_channels, 1), bias=False)
        self.spatial_norm = nn.BatchNorm2d(32)
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(32, 64, (1, 25), padding=(0, 12), bias=False)
        self.temporal_norm = nn.BatchNorm2d(64)
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Conv2d(64, d_model, (1, 1)),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Class token for temporal adaptability
        self.class_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Calculate sequence length after convolutions
        self.seq_len = self._get_seq_len(samples)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len + 1, d_model))
    
    def _get_seq_len(self, samples: int) -> int:
        """Calculate sequence length after convolutions"""
        # After temporal conv with padding (0, 12) and kernel (1, 25)
        return samples
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, 1, n_channels, samples]
        Returns:
            [batch_size, seq_len + 1, d_model]  # +1 for class token
        """
        batch_size = x.shape[0]
        
        # Spatial convolution
        x = self.spatial_conv(x)  # [B, 32, 1, samples]
        x = self.spatial_norm(x)
        x = F.gelu(x)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # [B, 64, 1, samples]
        x = self.temporal_norm(x)
        x = F.gelu(x)
        
        # Feature projection
        x = self.feature_proj(x)  # [B, d_model, 1, samples]
        
        # Reshape to sequence
        x = x.squeeze(2).transpose(1, 2)  # [B, samples, d_model]
        
        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        x = torch.cat([class_tokens, x], dim=1)  # [B, samples + 1, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1)]
        
        return x


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba Block for capturing temporal dependencies in both directions
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        
        self.config = config
        d_model = config.d_model
        d_inner = int(config.expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=config.bias)
        
        # Forward and backward Mamba blocks
        self.forward_mamba = MambaBlock(config)
        self.backward_mamba = MambaBlock(config)
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=config.bias)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        residual = x
        
        # Input projection
        x = self.in_proj(x)  # [B, L, d_inner * 2]
        x_forward, x_backward = x.chunk(2, dim=-1)  # Each [B, L, d_inner]
        
        # Forward processing
        forward_out = self.forward_mamba(x_forward)
        
        # Backward processing (reverse sequence)
        backward_out = self.backward_mamba(torch.flip(x_backward, [1]))
        backward_out = torch.flip(backward_out, [1])  # Flip back
        
        # Combine bidirectional outputs
        combined = forward_out + backward_out  # [B, L, d_inner]
        
        # Output projection
        output = self.out_proj(combined)  # [B, L, d_model]
        
        # Residual connection and normalization
        output = self.norm(output + residual)
        
        return output


class MambaBlock(nn.Module):
    """
    Core Mamba block with selective state space model
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        
        self.config = config
        d_model = config.d_model
        d_state = config.d_state
        d_conv = config.d_conv
        d_inner = int(config.expand * d_model)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=config.conv_bias,
            groups=d_inner,
            padding=d_conv - 1,
        )
        
        # Projections for selective scan
        self.x_proj = nn.Linear(d_inner, config.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, d_inner, bias=True)
        
        # State space parameters
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1))
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(d_inner))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_inner]
        Returns:
            [batch_size, seq_len, d_inner]
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Apply activation
        x = F.silu(x)
        
        # Convolution (need to transpose for conv1d)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        
        # Selective scan parameters
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        
        dt, B, C = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)
        
        # State space computation (simplified)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Simplified selective scan (in practice, this would use a more efficient implementation)
        y = self._selective_scan(x_conv, dt, A, B, C, self.D)
        
        return y
    
    def _selective_scan(self, u, delta, A, B, C, D):
        """
        Simplified selective scan implementation
        In practice, this would use optimized CUDA kernels
        """
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
        
        outputs = []
        for i in range(seq_len):
            # Discretization
            dA = torch.exp(delta[:, i:i+1].unsqueeze(-1) * A)  # [B, d_inner, d_state]
            dB = delta[:, i:i+1].unsqueeze(-1) * B[:, i:i+1].unsqueeze(1)  # [B, d_inner, d_state]
            
            # State update
            h = h * dA + dB * u[:, i:i+1].unsqueeze(-1)
            
            # Output
            y = torch.sum(h * C[:, i:i+1].unsqueeze(1), dim=-1)  # [B, d_inner]
            y = y + D * u[:, i]
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # [B, L, d_inner]


class TaskAwareMoE(nn.Module):
    """
    Task-aware Mixture of Experts with Universal Expert
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        num_classes: int,
        task_type: str,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.task_type = task_type
        
        # Task-specific experts
        self.experts = nn.ModuleList([
            ExpertBlock(d_model, dropout) for _ in range(num_experts)
        ])
        
        # Universal expert (always active)
        self.universal_expert = ExpertBlock(d_model, dropout)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_experts)
        )
        
        # Task embedding for routing
        self.task_embeddings = nn.Embedding(4, d_model)  # 4 task types
        self.task_mapping = {
            "motor_imagery": 0,
            "emotion": 1,
            "seizure": 2,
            "sleep": 3
        }
        
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            task_id: Optional task identifier
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get task embedding
        if task_id is None:
            task_id = self.task_mapping.get(self.task_type, 0)
        
        task_emb = self.task_embeddings(torch.tensor(task_id, device=x.device))
        task_emb = task_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Combine input with task embedding for gating
        gating_input = x + task_emb
        
        # Compute gating weights
        gate_logits = self.gate(gating_input.mean(dim=1))  # [B, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)  # [B, L, d_model]
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [B, L, num_experts, d_model]
        
        # Weight expert outputs
        gate_weights = gate_weights.unsqueeze(1).unsqueeze(-1)  # [B, 1, num_experts, 1]
        weighted_expert_output = torch.sum(expert_outputs * gate_weights, dim=2)  # [B, L, d_model]
        
        # Universal expert (always applied)
        universal_output = self.universal_expert(x)  # [B, L, d_model]
        
        # Combine task experts and universal expert
        output = 0.7 * weighted_expert_output + 0.3 * universal_output
        
        return output


class ExpertBlock(nn.Module):
    """Individual expert in the MoE"""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.expert = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.expert(x))


# =============== Factory Function ===============

def create_eegmamba(
    n_channels: int = 22,
    n_classes: int = 4,
    samples: int = 1000,
    task_type: str = "motor_imagery",
    num_experts: int = 4,
    num_mamba_layers: int = 6
) -> EEGMamba:
    """
    Create EEGMamba model
    
    Args:
        n_channels: Number of EEG channels
        n_classes: Number of classes
        samples: Number of time samples
        task_type: Type of EEG task
        num_experts: Number of experts in MoE
        num_mamba_layers: Number of Mamba layers
        
    Returns:
        EEGMamba model
    """
    config = MambaConfig(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2
    )
    
    return EEGMamba(
        n_channels=n_channels,
        n_classes=n_classes,
        samples=samples,
        config=config,
        num_experts=num_experts,
        num_mamba_layers=num_mamba_layers,
        task_type=task_type
    )


if __name__ == "__main__":
    # Test the model
    model = create_eegmamba(n_channels=22, n_classes=4, samples=1000)
    
    # Test input
    x = torch.randn(16, 1, 22, 1000)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different tasks
    print("\nTesting different task types:")
    tasks = ["motor_imagery", "emotion", "seizure", "sleep"]
    for task in tasks:
        model_test = create_eegmamba(task_type=task)
        out_test = model_test(x, task_id=None)
        print(f"Task: {task} -> Output shape: {out_test.shape}")
