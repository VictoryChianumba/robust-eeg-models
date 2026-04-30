import braindecode
print("Braindecode version:", braindecode.__version__)
from braindecode.models import EEGNetv4, Deep4Net, CTNet
from braindecode.models import CTNet
print("CTNet is available")

import torch
import numpy as np


import os
from collections import defaultdict


from utils.train_helpers import train_single_run, create_baseline_table
from models.eeg_mamba_fft import EEGMamba


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Baseline Run
seeds = [42, 123, 2024, 31415, 999]   

datasets = {
    "BNCIv2": ("BNCI2014001", 9),
    # Ended up not using DEAP. Can still use for further experiments 
    "DEAP": 32
}

dataset, n_subjects = datasets["BNCIv2"]
subjects = list(range(1, n_subjects+1))

SAVE_DIR = "results"          
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_CONFIGS = {
    'EEGNet': {
        'model_class': EEGNetv4,
        'training': {'lr': 1e-3, 'batch_size': 64, 'weight_decay': 1e-4, 'optimizer': torch.optim.Adam, 'scheduler': False}
    },
    'DeepConvNet': {
        'model_class': Deep4Net,
        'training': {'lr': 1e-3, 'batch_size': 64, 'weight_decay': 1e-4, 'optimizer': torch.optim.Adam, 'scheduler': False}
    },
    'CTNet': {
        'model_class': CTNet,
        'training':  {'lr': 1e-3, 'batch_size': 64, 'weight_decay': 1e-4, 'optimizer': torch.optim.AdamW, 'scheduler': False}
    },
    'EEGMamba': {
        'model_class': EEGMamba,
        'training': {'lr': 2e-4, 'batch_size': 128, 'weight_decay': 1e-6, 'optimizer': torch.optim.AdamW, 'scheduler': False}
    }
}

# Store all results: results[model_name][subject_id] = [acc1, acc2, acc3, acc4, acc5]
all_results = defaultdict(lambda: defaultdict(list))

for model_name in MODEL_CONFIGS.keys():
    print(f"RUNNING BASELINE FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    config = MODEL_CONFIGS[model_name]
    
    for subject_id in subjects:
        print(f"\n--- Subject {subject_id} ---")

        subject_scores = []
        for seed in seeds:
            print(f"  Seed {seed}: RUNNING")
            try:
                accuracy = train_single_run(model_name, subject_id, seed, dataset, config, device)
                subject_scores.append(accuracy)
                print(f"  Seed {seed}: {accuracy:.4f}")
            except Exception as e:
                print(f"  Seed {seed}: FAILED ({e})")
                subject_scores.append(np.nan)

        # Store results for this (model, subject) pair
        all_results[model_name][subject_id] = subject_scores

        # Calculate stats for this subject
        valid_scores = [s for s in subject_scores if not np.isnan(s)]
        if valid_scores:
            mean_acc = np.mean(valid_scores)
            std_acc = np.std(valid_scores)
            print(f"  Subject {subject_id} baseline: {mean_acc:.4f} ± {std_acc:.4f}")
        else:
            print(f"  Subject {subject_id}: ALL RUNS FAILED")

# Create and display results
baseline_df = create_baseline_table(all_results, subjects)
print(f"\n{'='*80}")
print("FINAL BASELINE RESULTS")
print(f"{'='*80}")

# Subject-wise baselines
for model_name in MODEL_CONFIGS.keys():
    print(f"\n{model_name}:")
    model_data = baseline_df[baseline_df['Model'] == model_name]

    subject_means = []
    for _, row in model_data.iterrows():
        if not np.isnan(row['Mean_Accuracy']):
            print(f"  Subject {row['Subject']}: {row['Mean_Accuracy']:.4f} ± {row['Std_Accuracy']:.4f}")
            subject_means.append(row['Mean_Accuracy'])
        else:
            print(f"  Subject {row['Subject']}: FAILED")

    # Dataset-wide average
    if subject_means:
        dataset_mean = np.mean(subject_means)
        dataset_std = np.std(subject_means)
        print(f"  → Dataset average: {dataset_mean:.4f} ± {dataset_std:.4f}")
    else:
        print(f"  → Dataset average: FAILED")

# Save results
baseline_df.to_csv('baseline_results.csv', index=False)
print(f"\nResults saved to baseline_results.csv")
