import torch
import torch.nn as nn 
import torch.nn.functional as F 

class DeepConvNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=1125):
        super(DeepConvNet, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 25, (1, 5)),
            nn.Conv2d(25, 25, (channels, 1)),
            nn.BatchNorm2d(25), 
            nn.ELU(), 
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
            
            nn.Conv2d(25, 50, (1, 5)),
            nn.BatchNorm2d(50), 
            nn.ELU(), 
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
            
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100), 
            nn.ELU(), 
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
            
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200), 
            nn.ELU(), 
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.5),
        )
        
        # Use dummy input to infer flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            flat_dim = self.conv_block(dummy).view(1, -1).shape[1]

        self.classify = nn.Linear(flat_dim, num_classes)
        
        #self.classify = nn.Linear(200 * (samples // (2 * 4)), num_classes)  
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)   
        return self.classify(x)
    