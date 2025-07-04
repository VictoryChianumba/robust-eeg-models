import torch
import torch.nn as nn
import math

class MultiScaleEEGTransformer(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001, 
                 d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        self.channels = channels
        self.d_model = d_model
        self.patch_sizes = [10, 25, 50]
        
        # Store components separately
        self.patch_embeds = nn.ModuleList()
        self.transformers = nn.ModuleList()
        self.pos_embeds = nn.ParameterList()  # Use ParameterList for tensors
        
        for patch_size in self.patch_sizes:
            num_patches = samples // patch_size
            
            # Patch embedding
            patch_embed = nn.Linear(channels * patch_size, d_model)
            self.patch_embeds.append(patch_embed)
            
            # Positional encoding (Parameter, not regular tensor)
            pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))
            self.pos_embeds.append(pos_embed)
            
            # Transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True
            )
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.transformers.append(transformer)
        
        # Store num_patches for each scale
        self.num_patches_list = [samples // ps for ps in self.patch_sizes]
        
        # Fusion layer
        self.fusion = nn.Linear(d_model * len(self.patch_sizes), d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze(1)  # (batch, channels, samples)
        
        scale_features = []
        
        for i, patch_size in enumerate(self.patch_sizes):
            # Create patches for this scale
            patches = x.unfold(2, patch_size, patch_size)
            patches = patches.transpose(1, 2)
            patches = patches.reshape(batch_size, -1, self.channels * patch_size)
            
            # Process with corresponding components
            embedded = self.patch_embeds[i](patches)
            embedded = embedded + self.pos_embeds[i]
            encoded = self.transformers[i](embedded)
            
            # Global pooling
            pooled = encoded.mean(dim=1)
            scale_features.append(pooled)
        
        # Concatenate multi-scale features
        fused = torch.cat(scale_features, dim=1)
        fused = self.fusion(fused)
        
        return self.classifier(fused)