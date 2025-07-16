
import torch
import torch.nn as nn

class DeepConvNet(nn.Module): 
    """
    Deep ConvNet from Schirrmeister et al. (2017)
    Source: "Deep learning with convolutional neural networks for EEG decoding and visualization"
    """
    
    def __init__(self, num_classes=4, channels=22, samples=1125, dropout_rate=0.5):
        super().__init__()
        
        # First convolution block
        self.conv_temporal = nn.Conv2d(1, 25, (1, 10), bias=True)
        self.conv_spatial = nn.Conv2d(25, 25, (channels, 1), bias=True)
        self.bnorm1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(25, 50, (1, 10), bias=True)
        self.bnorm2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third convolution block
        self.conv3 = nn.Conv2d(50, 100, (1, 10), bias=True)
        self.bnorm3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth convolution block
        self.conv4 = nn.Conv2d(100, 200, (1, 10), bias=True)
        self.bnorm4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1)
        self.elu4 = nn.ELU()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Calculate feature dimension
        self._get_feature_dim(channels, samples)
        
        # Classification layer
        self.classify = nn.Linear(self.feature_dim, num_classes)
    
    def _get_feature_dim(self, channels, samples):
        """Calculate the dimension after convolutions"""
        # Simulate forward pass to get dimensions
        with torch.no_grad():
            x = torch.zeros(1, 1, channels, samples)
            x = self._forward_features(x)
            self.feature_dim = x.view(1, -1).size(1)
    
    def _forward_features(self, x):
        """Forward pass through feature extraction layers"""
        x = self.conv_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.elu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        return x
    
    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)
