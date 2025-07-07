

import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001):
        super(DeepConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=(0, 2))
        self.batchnorm1 = nn.BatchNorm2d(25)

        self.conv2 = nn.Conv2d(25, 25, (channels, 1))
        self.batchnorm2 = nn.BatchNorm2d(25)
        self.elu = nn.ELU()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(25, 50, (1, 5), padding=(0, 2))
        self.batchnorm3 = nn.BatchNorm2d(50)

        self.conv4 = nn.Conv2d(50, 100, (1, 5), padding=(0, 2))
        self.batchnorm4 = nn.BatchNorm2d(100)

        self.conv5 = nn.Conv2d(100, 200, (1, 5), padding=(0, 2))
        self.batchnorm5 = nn.BatchNorm2d(200)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate the input size for the linear layer dynamically
        # We need a dummy forward pass to determine this
        dummy_input = torch.randn(1, 1, channels, samples)
        dummy_output = self._forward_conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).size(1)

        self.classify = nn.Linear(flattened_size, num_classes)

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        # x = self.batchnorm3(x)
        # x = self.elu(x)
        # x = self.pool(x)
        # x = self.dropout(x)
        # x = self.conv4(x)
        # x = self.batchnorm4(x)
        # x = self.elu(x)
        # x = self.pool(x)
        # x = self.dropout(x)
        # x = self.conv5(x)
        # x = self.batchnorm5(x)
        # x = self.elu(x)
        # x = self.pool(x)
        # x = self.dropout(x)
        x = self.globalpool(x)
        return x


    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)
