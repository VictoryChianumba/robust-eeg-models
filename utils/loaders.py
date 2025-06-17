import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
    def __init__(self, subject_ids, data_dir="data/processed", transform=None):
        """_summary_

        Args:
            subject_ids (_type_): List of subject numbers (1-9)
            data_dir (str, optional): Directory with preprocesseed .npy files. Defaults to "data/processed".
            transform (_type_, optional): Optional transform to apply to each sample. Defaults to None.
            
        
        """
        
        self.X = []
        self.y = []
        self.transform = transform
        for sid in subject_ids:
            X_path = os.path.join(data_dir, f"X_subject{sid}.npy")
            y_path = os.path.join(data_dir, f"y_subject{sid}.npy")
            
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                raise FileNotFoundError(f"Missing data for subject {sid} in {data_dir}")
            
            X_subj = np.load(X_path)         # shape: (trials, channels, time)
            y_subj = np.load(y_path)         # shape: (trials,)    
            self.X.append(X_subj)
            self.y.append(y_subj)
            
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        # print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")
        # print(f"X dtype: {self.X.dtype}, y dtype: {self.y.dtype}")
        # print(f"y unique: {np.unique(self.y)}")

        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        # we convert to (1, channels, time) for 2D CNN input  
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
# def get_dataloader(subject_ids, batch_size=64, shuffle=True, data_dir="data/processed"):
#     dataset = EEGDataset(subject_ids=subject_ids, data_dir = data_dir)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return loader
def get_dataloader(subject_ids, batch_size=32, shuffle=True, data_dir='data/processed', return_dataset=False):
    dataset = EEGDataset(subject_ids=subject_ids, data_dir=data_dir)
    if return_dataset:
        return dataset  # <-- this is the missing part
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
