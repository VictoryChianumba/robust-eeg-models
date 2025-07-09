import optuna
import pickle
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from training.train_model import train_model
from models.eegnet import EEGNet
from models.deepconvnet import DeepConvNet

def create_and_train_model(trial, train_loader, val_loader, model_type='EEGNet', 
                          num_classes=4, channels=22, samples=1125, epochs=50):
    """
    Clean objective function that only handles model creation and hyperparameter suggestions
    """
    # Hyperparameter suggestions
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Model-specific parameters
    if model_type == 'EEGNet':
        F1 = trial.suggest_categorical('F1', [8, 16, 32])
        D = trial.suggest_categorical('D', [2, 4, 8])
        model = EEGNet(num_classes=num_classes, channels=channels, samples=samples,
                      F1=F1, D=D, dropout=dropout)
    elif model_type == 'DeepConvNet':
        model = DeepConvNet(num_classes=num_classes, channels=channels, samples=samples, 
                           dropout=dropout)
    
    # Train the model using your existing training function
    best_val_acc = train_model(model, train_loader, val_loader, lr, weight_decay, epochs, trial)
    
    return best_val_acc

def save_study_results(study, filename_prefix):
    """Save study results in multiple formats"""
    
    # Save the complete study object
    with open(f'{filename_prefix}_study.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    # Save best results as JSON
    results = {
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials)
    }
    
    with open(f'{filename_prefix}_best_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Study saved as {filename_prefix}_study.pkl and {filename_prefix}_best_results.json")

def run_overnight_optimization(X, y, model_type='EEGNet', max_hours=8):
    """
    Run optimization with your data
    """
    # Data preparation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # Model parameters
    num_classes = len(np.unique(y))
    channels = X.shape[2]
    samples = X.shape[3]
    
    def objective(trial):
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return create_and_train_model(
            trial, train_loader, val_loader, model_type, 
            num_classes, channels, samples, epochs=50  # Reduced for hyperparameter search
        )
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    # Callback to save progress
    def save_callback(study, trial):
        if trial.number % 5 == 0:  # Save every 5 trials
            save_study_results(study, f'{model_type}_trial_{trial.number}')
            print(f"💾 Progress saved at trial {trial.number}")
            print(f"🏆 Current best: {study.best_trial.value:.4f}")
    
    # Run optimization
    timeout_seconds = max_hours * 3600
    
    try:
        study.optimize(
            objective, 
            n_trials=1000,  # Large number, will stop due to timeout
            timeout=timeout_seconds,
            callbacks=[save_callback]
        )
    except KeyboardInterrupt:
        print("⏹️  Manually stopped optimization")
    finally:
        save_study_results(study, f'{model_type}_final')
        print(f"🌅 Optimization complete! {len(study.trials)} trials completed")
        print(f"🏆 Best result: {study.best_trial.value:.4f}")
        print(f"📋 Best params: {study.best_trial.params}")
    
    return study
