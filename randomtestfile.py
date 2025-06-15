import torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.randn(100, 22, 1001)  # EEG samples
y = torch.randint(0, 4, (100,))  # class labels

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(loader.shape)

for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
    break
