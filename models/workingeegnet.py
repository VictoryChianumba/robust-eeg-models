
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1125, F1=8, D=2, dropout=0.25):
        super(EEGNet, self).__init__()
        
        self.activations = {}

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1*D, (channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        self.classify = nn.Linear((F1*D) * ((samples // 32)), num_classes)

    def forward(self, x):
        x = self.firstconv(x)
        self.activations['firstconv'] = x.detach().cpu()
        x = self.depthwiseConv(x)
        self.activations['depthwiseConv'] = x.detach().cpu()
        x = self.separableConv(x)
        self.activations['separableConv'] = x.detach().cpu()

        x = x.view(x.size(0), -1)  # flatten
        return self.classify(x)
