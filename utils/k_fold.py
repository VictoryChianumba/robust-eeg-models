import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def k_fold_cross_validation(X, y, model_class, k_folds=5, epochs=150, batch_size=32, device='cpu'):
    """
    Perform k-fold cross-validation
    
    Args:
        X: Input data (numpy array)
        y: Labels (numpy array) 
        model_class: Function that returns a fresh model instance
        k_folds: Number of folds
        epochs: Number of epochs per fold
        batch_size: Batch size
        device: Device to train on
    
    Returns:
        Dictionary with results
    """
    
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    
    print(f"Starting {k_folds}-Fold Cross-Validation")
    print("="*50)
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\nFOLD {fold + 1}/{k_folds}")
        print("-" * 20)
        
        # Create data loaders for this fold
        train_dataset = TensorDataset(
            torch.tensor(X[train_idx], dtype=torch.float32),
            torch.tensor(y[train_idx], dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X[val_idx], dtype=torch.float32), 
            torch.tensor(y[val_idx], dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
        
        # Fresh model for this fold
        model = model_class().to(device)
        # optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

       
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # ADD THIS: Capture initial weights for debugging
        initial_weight = None
        if hasattr(model, 'conv_block') and len(model.conv_block) > 0:
            # For DeepConvNet
            initial_weight = model.conv_block[0].weight.clone().detach()
        elif hasattr(model, 'firstconv'):
            # For EEGNet  
            initial_weight = model.firstconv[0].weight.clone().detach()
        else:
            print("⚠️ Cannot find conv layer for weight debugging")
        
        
        # Training for this fold
        best_val_acc = 0
        patience_counter = 0
        patience = 30
        
        for epoch in range(epochs):
            # Training
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                
                # debug, checking for vanishing gradients
                total_grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5

                print(f"Gradient norm: {total_grad_norm}")

                if total_grad_norm < 1e-6:
                    print("⚠️ WARNING: Vanishing gradients!")
                
                optimizer.step()
                running_loss += loss.item()
            
            train_loss = running_loss / len(train_loader)
            
            # ADD THIS: Check weight changes after first epoch
            if epoch == 0 and initial_weight is not None:
                if hasattr(model, 'conv_block') and len(model.conv_block) > 0:
                    current_weight = model.conv_block[0].weight
                elif hasattr(model, 'firstconv'):
                    current_weight = model.firstconv[0].weight
                
                weight_change = torch.sum(torch.abs(current_weight - initial_weight))
                print(f"Fold {fold+1} - Weight change after epoch 1: {weight_change.item():.8f}")
                
                if weight_change < 1e-6:
                    print("⚠️ WARNING: Weights are barely changing!")
                else:
                    print("✅ Weights are updating normally")
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            val_accuracy = correct / total
            # scheduler.step(val_accuracy)
            
            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # if epoch % 5 == 0:  # Print every 5 epochs instead of 20
            #     print(f"Fold {fold+1} - Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_accuracy:.4f}")
            
            print(f"Epoch {epoch}: Loss={train_loss:.6f}, Val Acc={val_accuracy:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")

        
        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold + 1} Best Accuracy: {best_val_acc:.4f}")
    
    # Final results
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for i, acc in enumerate(fold_accuracies):
        print(f"Fold {i+1}: {acc:.4f}")
    print(f"Mean: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Range: {min(fold_accuracies):.4f} - {max(fold_accuracies):.4f}")
    
    return {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_accuracy,
        'std_accuracy': std_accuracy
    }