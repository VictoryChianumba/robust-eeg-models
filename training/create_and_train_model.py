import optuna
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

# Run optimization
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)

study.optimize(create_and_train_model(), n_trials=100, timeout=3600*8)  # 8 hours

# Results
print(f"Best trial: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_params}")

# Visualization
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
