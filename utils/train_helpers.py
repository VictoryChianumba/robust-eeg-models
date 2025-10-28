from repr import repr_helpers as rp
import numpy as np
import torch
from utils import load_subject as ls
from braindecode import EEGClassifier
from skorch.helper import predefined_split
import pandas as pd

def train_single_run(model_name, subject_id, seed, dataset, config, device):

    # 0. RNG reproducibility ---------------------------------------------------

    rp.set_all_seeds(seed)
    rng_state_np    = np.random.get_state()
    rng_state_torch = torch.get_rng_state()
    env_fp          = rp.get_environment_fingerprint()

    print(f"\n=== Processing Subject {subject_id} for seed {seed} ===")

    # 1. data ------------------------------------------------------------------
    # Load data
    train_set, test_set, train_subset, val_subset, adv  = ls.load_subject_data_cached(dataset, subject_id)

    # Build simple tensors to compute stats on train windows only
    X_train = np.stack([train_set[i][0] for i in range(len(train_set))])  # (N,C,T)
    train_mean = X_train.mean(axis=(0,2), keepdims=True)  # (1,C,1)
    train_std  = X_train.std(axis=(0,2), keepdims=True) + 1e-6

    # Empirical bounds for later clipping in attack
    train_min = X_train.min(axis=(0,2), keepdims=True)
    train_max = X_train.max(axis=(0,2), keepdims=True)

    # expose indices explicitly
    train_idx = train_subset.indices
    val_idx   = val_subset.indices
    test_idx  = np.arange(len(test_set))

    # 2. model & config --------------------------------------------------------


    # Extract model params from dataset, initialise model and set hyper-parameters
    # classes needed for clf
    classes = torch.unique(torch.tensor([sample[1] for sample in train_subset])).tolist()
    n_classes = len(classes)
    n_channels = train_subset[0][0].shape[0]
    n_times = train_subset[0][0].shape[1]

    model = config['model_class'](
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
    )

    # Special handling for EEGMamba
    if model_name == 'EEGMamba':
        model.enable_moe(False)  # Use standard classifier for baseline. Paper explicitly removes moe modules for
                                # single use mamba
        # Enable MoE head instead of standard classifier (For multi-use only)
        model.use_moe = False

    # 4. fit -------------------------------------------------------------------

    # Create new classifier with best parameters
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        train_split=predefined_split(val_subset),  # Use all training data
        optimizer=config['training']['optimizer'],
        optimizer__lr=config['training']['lr'],
        optimizer__weight_decay=config['training']['weight_decay'],
        batch_size=config['training']['batch_size'],
        callbacks=["accuracy"],
        device=device,
        classes=classes,
        max_epochs=500,
    )

    # Train on full training set
    clf.fit(train_subset, y=None)

    # 5. test accuracy ---------------------------------------------------------

    # Evaluate the model after training
    y_test = test_set.get_metadata().target
    test_accuracy = clf.score(test_set, y = y_test)

    # 6. save everything -------------------------------------------------------

    rp._save_run(model_name, subject_id, seed,
              clf, test_set, rng_state_np, rng_state_torch,
              train_idx, val_idx, test_idx,
              train_mean, train_std, train_min, train_max,
              config, env_fp, device)

    return test_accuracy

# ==============================================================================

# Generate final baseline table
def create_baseline_table(results, subjects: list):
    """Create a nice table of baselines"""
    rows = []

    for model_name in results.keys():
        for subject_id in subjects:
            scores = results[model_name][subject_id]
            valid_scores = [s for s in scores if not np.isnan(s)]

            if valid_scores:
                mean_acc = np.mean(valid_scores)
                std_acc = np.std(valid_scores)
                n_valid = len(valid_scores)
            else:
                mean_acc = std_acc = n_valid = np.nan

            rows.append({
                'Model': model_name,
                'Subject': subject_id,
                'Mean_Accuracy': mean_acc,
                'Std_Accuracy': std_acc,
                'N_Valid_Runs': n_valid,
                'Individual_Scores': scores
            })

    return pd.DataFrame(rows)
