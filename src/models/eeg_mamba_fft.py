import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from braindecode.models.base import EEGModuleMixin

# ------------------------------------------------------------------
# 1. FFT-Based Mamba Implementation (No external dependencies)
# ------------------------------------------------------------------
class FFTMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, bidirectional: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        self.d_state = d_state
        self.bidir   = bidirectional
        self.A_log   = nn.Parameter(torch.randn(d_state) * 0.02)
        self.B_proj  = nn.Linear(d_model, d_state)
        self.C_proj  = nn.Linear(d_state, d_model)
        self.D       = nn.Parameter(torch.ones(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        A = -torch.exp(self.A_log).unsqueeze(0)
        u = self.B_proj(x)

        n_fft = 2 ** (T - 1).bit_length()
        t = torch.arange(T, dtype=torch.float, device=x.device)
        K = torch.exp(A * t.unsqueeze(1)).to(torch.float32)
        if self.bidir:
            K = K + torch.flip(K, dims=[0])

        K_f = fft.rfft(K, n=n_fft, dim=0)
        u_f = fft.rfft(u, n=n_fft, dim=1)
        y_f = K_f.unsqueeze(0) * u_f
        y = fft.irfft(y_f, n=n_fft, dim=1)[..., :T, :]

        out = self.dropout(self.C_proj(y)) + self.D * x
        return out

class SpatialDW(nn.Module):
    """
    Spatial 1×1 conv (C → D)  followed by depth-wise 1-D temporal conv.
    Keeps (B, C, T) → (B, T, D).
    """
    def __init__(self, n_chans: int, d_model: int, kernel: int = 15, dropout: float = 0.0):
        super().__init__()
        self.spatial = nn.Conv2d(1, d_model, (n_chans, 1), bias=False)   # (B,1,C,T)→(B,D,1,T)
        self.temporal = nn.Conv1d(
            d_model, d_model, kernel_size=kernel,
            padding=kernel//2, groups=d_model, bias=False
        )
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):               # (B, C, T)
        x = x.unsqueeze(1)              # (B,1,C,T)
        x = self.spatial(x).squeeze(2)  # (B,D,T)
        x = self.temporal(x)            # (B,D,T)
        x = x.transpose(1, 2)           # (B,T,D)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)   # (B,T+1,D)

class BiMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.1, ffn_mult=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = FFTMamba(d_model, d_state=d_state, bidirectional=True, dropout=dropout)

        # depth-wise temporal conv
        self.temporal = nn.Conv1d(
            d_model, d_model, kernel_size=15, padding=7, groups=d_model, bias=False
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x):                # (B, T, D)
        x = x + self.mamba(self.norm1(x))
        # depth-wise conv
        y = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x

class TaskMoE(nn.Module):
    """Task-aware Mixture-of-Experts head."""
    def __init__(self, d_model: int, n_classes: int,
                 n_experts: int = 9, k: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(d_model, n_classes) for _ in range(n_experts)
        ])
        self.gate = nn.Linear(d_model, n_experts)
        self.k = k

    def forward(self, x):
      logits = self.gate(x)
      probs = F.softmax(logits, dim=-1)
      topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)

      # Compute all expert outputs
      expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # (B, n_experts, n_classes)
      
      # Select top-k experts using advanced indexing
      batch_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1)  # (B, 1)
      selected_outputs = expert_outputs[batch_indices, topk_idx]  # (B, k, n_classes)
      
      # Weight and sum
      weights = topk_vals.unsqueeze(-1)  # (B, k, 1)
      y = (weights * selected_outputs).sum(dim=1)  # (B, n_classes)
      
      return y
# ------------------------------------------------------------------
# 2. Fixed Braindecode Wrapper
# ------------------------------------------------------------------
class EEGMamba(EEGModuleMixin, nn.Module):
    """Braindecode-compatible EEGMamba model."""
    
    def __init__(
        self,
        # Braindecode standard parameters
        n_chans=None,
        n_outputs=None,
        n_times=None,
        chs_info=None,
        input_window_seconds=None,
        sfreq=None,
        # EEGMamba specific parameters
        d_model=128,
        n_layers=8,
        d_state=64,          # bigger state
        dropout=0.3,         # overall dropout
        n_experts=9,
        k=2,
        # Backward compatibility aliases
        in_chans=None,
        n_classes=None,
        input_window_samples=None,
    ):
        
        # Initialize base class
        super().__init__(
            n_outputs=n_outputs,
            n_chans=n_chans,
            chs_info=chs_info,
            n_times=n_times,
            input_window_seconds=input_window_seconds,
            sfreq=sfreq,
        )
        
        # Store model parameters
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.k = k
        self.d_state = d_state
        self.dropout = dropout
        
        
        # Build the model
        self.st_dw = SpatialDW(self.n_chans, d_model)
        self.layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Simple classification head (no MoE for standard compatibility)
        self.classifier = nn.Linear(d_model, self.n_outputs)
        
        # Optional: MoE head (use when you want advanced features)
        self.moe_head = TaskMoE(d_model, self.n_outputs, n_experts, k)
        self.use_moe = False  # Toggle this for MoE vs standard mode
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        """
        Forward pass compatible with Braindecode.
        
        Args:
            x: Input tensor [batch_size, n_chans, n_times]
            
        Returns:
            logits: Output tensor [batch_size, n_outputs]
        """
        # Spatial-temporal adaptive processing
        x = self.st_dw(x)  # (B, T+1, D)
        
        # Bidirectional Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Use class token
        x = self.norm(x).mean(dim=1)   # (B, D)
        # x = self.norm(x[:, 0])
        
        # Classification
        if self.use_moe:
            return self.moe_head(x)  # Only expects tensor
        else:
            return self.classifier(x)  # Only returns tensor

    def get_output_shape(self):
        """Required method for Braindecode compatibility."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.n_chans, self.n_times)
            output = self.forward(dummy_input)
            return output.shape
    
    def enable_moe(self, enable=True):
        """Enable/disable MoE head."""
        self.use_moe = enable
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# ------------------------------------------------------------------
# 3. Factory Function for Easy Usage
# ------------------------------------------------------------------
def create_eegmamba(n_chans, n_outputs, n_times, **kwargs):
    """
    Factory function to create EEGMamba model.
    
    Args:
        n_chans: Number of EEG channels
        n_outputs: Number of classes
        n_times: Number of time samples
        **kwargs: Additional model parameters
        
    Returns:
        EEGMamba model instance
    """
    return EEGMamba(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        **kwargs
    )
