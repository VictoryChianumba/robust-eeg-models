import torch
import optuna
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from training.create_and_train_model import create_and_train_model
from training.save_study_results import save_study_results

def run_optimization(X, y, model_type='EEGNet', n_trials=50, save_every=5):

    # Data preparation (done once)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets (done once)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # Model parameters
    num_classes = len(np.unique(y))
    channels = X.shape[2]
    samples = X.shape[3]
    
    def objective(trial):
        # Only batch_size is suggested here since it affects data loaders
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Create loaders with trial-specific batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return create_and_train_model(
            trial, train_loader, val_loader, model_type, 
            num_classes, channels, samples
        )

    # Create study with database storage
    study = optuna.create_study(
        direction='maximize',
        study_name=f'{model_type}_optimization',
        storage=f'sqlite:///{model_type}_study.db',  # Saves to database
        load_if_exists=True  # Resume if exists
    )
    
    # Callback to save every few trials
    def save_callback(study, trial):
        if trial.number % save_every == 0:
            save_study_results(study, f'{model_type}_trial_{trial.number}')
            print(f"💾 Progress saved at trial {trial.number}")
    
    
    study.optimize(objective, n_trials=n_trials)

    # Run optimization with callback
    study.optimize(objective, n_trials=n_trials, callbacks=[save_callback])
    
    # Final save
    save_study_results(study, f'{model_type}_final')
    
    return study
