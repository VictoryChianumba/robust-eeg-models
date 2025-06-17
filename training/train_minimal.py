import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from models.eegnet import EEGNet  # adjust if necessary
from utils.loaders import get_dataloader
from utils.checks import check_shapes
from utils.preprocessing import preprocess_all_subjects

# ---------- Config ----------
subject_id = 8
batch_size = 32
epochs = 50
lr = 0.001
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ----------Re run the Preprocessing --------
preprocess_all_subjects(subject_ids=[1, 2, 3, 4, 5, 6, 7])

# ---------- Load Data ----------
dataset = get_dataloader(subject_ids=[1,2,3,4,5,6,7], return_dataset=True)
X, y = dataset.X, dataset.y
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
print(f"y unique: {np.unique(y)}")
check_shapes(X, y)

# Ensure y is a NumPy array
y = np.array(y)

# Filter out any invalid samples (e.g., negative class labels)
valid_mask = y >= 0
X = X[valid_mask]
y = y[valid_mask]

# Debug: Check what labels look like
print("Original labels:", np.unique(y), "min:", y.min(), "max:", y.max())

# Remap labels to start from 0
unique_labels = np.unique(y)
label_map = {old: new for new, old in enumerate(unique_labels)}
y = np.vectorize(label_map.get)(y)

# Sanity check
print("Cleaned labels:", np.unique(y), "min:", y.min(), "max:", y.max())
# ------------- Train test split -------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Toy dataset
X_small, y_small = [], []
for cls in range(4):
    idx = np.where(y == cls)[0][:30]
    X_small.append(X[idx])
    y_small.append(y[idx])

X_small = np.concatenate(X_small, axis=0)
y_small = np.concatenate(y_small, axis=0)

X_train = np.expand_dims(X_train, axis=1)    # from (N, 22, 1001) → (N, 1, 22, 1001)
X_val = np.expand_dims(X_val, axis=1)        # from (N, 22, 1001) → (N, 1, 22, 1001)

# asserting correct dimensions. good for debugging
assert X_train.shape[1] == 1, "Expected dummy input channel dimension (1)"
assert X_train.shape[2] == 22, "Expected 22 EEG channels"
assert X_train.shape[3] == 1001, "Expected 1001 time samples"
print(f"X shape: {X_small.shape}, y shape: {y_small.shape}")


# wrap in Dataloader
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ---------- Model Setup ----------
model = EEGNet(num_classes=len(torch.unique(torch.tensor(y))), channels=X.shape[1], samples=X.shape[2] ).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------- Training Loop ----------
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device).float()
        batch_y = batch_y.to(device).long()
        

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss:.4f}")
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.to(device))
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y.to(device)).sum().item()
            total += batch_y.size(0)
    val_acc = correct / total

    print(f"Epoch [{epoch}/5] - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

print("✅ Training complete.")

# --- Evaluation ---
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\n✅ Final Val Accuracy: {acc:.4f}")

print("Prediction distribution:", np.bincount(all_preds))
print(outputs[:5])  # see if they're all biased to class 0
probs = torch.softmax(outputs, dim=1)
print(probs[0])  # e.g., tensor([0.99, 0.002, 0.002, 0.001])


print("\n🧾 Classification Report:")
print(classification_report(all_labels, all_preds))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))