
import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001, dropout=0.5):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 25, kernel_size=(1, 5), padding=(0, 2))
        self.batchnorm1 = nn.BatchNorm2d(25)
        self.elu = nn.ELU()

        self.conv2 = nn.Conv2d(25, 25, kernel_size=(channels, 1))
        self.batchnorm2 = nn.BatchNorm2d(25)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout_layer = nn.Dropout(dropout)

        self.conv3 = nn.Conv2d(25, 50, kernel_size=(1, 5), padding=(0, 2))
        self.batchnorm3 = nn.BatchNorm2d(50)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))

        self.features = nn.Sequential(
            self.conv1,
            self.batchnorm1,
            self.conv2,
            self.batchnorm2,
            self.elu,
            self.pool,
            self.dropout_layer,
            self.conv3,
            self.batchnorm3,
            self.elu,
            self.pool,
            self.dropout_layer
        )
   
        # self.classify = nn.Linear(50, num_classes)

        self.classify = nn.Linear(50 , num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.globalpool(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)
