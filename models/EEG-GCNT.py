import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class EEGGCNTransformer(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        
        self.channels = channels
        self.d_model = d_model
        
        # EEG electrode adjacency matrix (simplified - you'd use real electrode positions)
        self.register_buffer('adj_matrix', self._create_eeg_adjacency())
        
        # Graph convolution layers
        self.gcn1 = GraphConv(samples, d_model // 2)
        self.gcn2 = GraphConv(d_model // 2, d_model)
        
        # Positional encoding for channels
        self.pos_embed = nn.Parameter(torch.randn(1, channels, d_model))
        
        # Transformer for spatial-temporal interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes)
        )
        
    def _create_eeg_adjacency(self):
        """Create simplified EEG electrode adjacency matrix"""
        # This is simplified - in practice you'd use real electrode coordinates
        adj = torch.eye(self.channels)
        
        # Add connections between neighboring electrodes (simplified)
        for i in range(self.channels - 1):
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1
        
        # Add some cross-hemisphere connections
        if self.channels >= 20:
            adj[0, 10] = adj[10, 0] = 1  # Connect left-right
            adj[5, 15] = adj[15, 5] = 1
        
        # Normalize adjacency matrix
        degree = adj.sum(dim=1, keepdim=True)
        adj = adj / (degree + 1e-6)
        
        return adj
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze(1)  # (batch, channels, samples)
        
        # Graph convolution
        x = F.relu(self.gcn1(x, self.adj_matrix))
        x = F.relu(self.gcn2(x, self.adj_matrix))  # (batch, channels, d_model)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer processing
        x = self.transformer(x)  # (batch, channels, d_model)
        
        # Global pooling across channels
        x = x.mean(dim=1)  # (batch, d_model)
        
        return self.classifier(x)

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):
        # x: (batch, nodes, features)
        # adj: (nodes, nodes)
        support = torch.matmul(x, self.weight)  # (batch, nodes, out_features)
        output = torch.matmul(adj, support.transpose(0, 1)).transpose(0, 1)  # Apply adjacency
        return output + self.bias