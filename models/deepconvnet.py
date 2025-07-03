import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1001):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=1, padding=0),
            nn.Conv2d(25, 25, kernel_size=(channels, 1), stride=1),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, kernel_size=(1,5), stride=1, padding=0),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, kernel_size=(1,5), stride=1, padding=0),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, kernel_size=(1,5), stride=1, padding=0),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.5),
        )

        # compute the flattened dimension after 4 blocks
        dummy = torch.zeros(1, 1, channels, samples)
        out = self.features(dummy)
        flattened = out.view(1, -1).shape[1]

        self.classify = nn.Linear(flattened, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classify(x)
