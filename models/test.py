from eegnet import EEGNet
from deepconvnet import DeepConvNet
import torch

x = torch.randn(8, 1, 22, 1125)  # (batch, channels=1, EEG channels, time)

model = EEGNet()
print(model(x).shape)  # Should output: (8, 4)

model = DeepConvNet()
print(model(x).shape)  # Should output: (8, 4)
