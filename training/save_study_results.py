


import pickle
import json

def save_study_results(study, filename_prefix):
    """Save study results in multiple formats for safety"""
    
    # Save the complete study object (can resume from this)
    with open(f'{filename_prefix}_study.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    # Save best results as JSON (human readable)
    results = {
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': len(study.trials),
        'study_name': study.study_name
    }
    
    with open(f'{filename_prefix}_best_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save all trial data as CSV (for analysis)
    import pandas as pd
    trials_df = study.trials_dataframe()
    trials_df.to_csv(f'{filename_prefix}_all_trials.csv', index=False)
    
    print(f"✅ Study saved as:")
    print(f"  - {filename_prefix}_study.pkl (full study)")
    print(f"  - {filename_prefix}_best_results.json (best config)")
    print(f"  - {filename_prefix}_all_trials.csv (all trials)")
