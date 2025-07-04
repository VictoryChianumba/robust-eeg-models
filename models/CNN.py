import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # Use 1D kernels appropriate for EEG
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2))  # Temporal conv
        self.pool = nn.AvgPool2d((1, 2))  # Only pool in time dimension
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 5))  # Spatial conv (all channels)
        
        # Calculate flattened size
        with torch.no_grad():
            x = torch.zeros(1, 1, *input_shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.numel()

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

