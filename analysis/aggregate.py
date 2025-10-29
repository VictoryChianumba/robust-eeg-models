import pandas as pd
import numpy as np

def aggregate_across_seeds(df_clean):
    """
    Simple aggregation across seeds for experimental conditions
    """
    print("SEED AGGREGATION")
    print(f"Input dataset shape: {df_clean.shape}")

    # Define grouping variables
    grouping_vars = ['subject_id', 'model_name', 'attack', 'architecture_type', 'muV_budget']

    # For numeric columns, calculate mean and std
    numeric_vars = [
        'ASR', 'spearman_IG', 'clean_acc', 'snr_db_mean', 'snr_db_std',
        'median_L2_success', 'mean_L2_all', 'acc_drop', 'rel_acc_drop'
    ]

    # Filter to only include columns that actually exist in the dataframe
    numeric_vars = [var for var in numeric_vars if var in df_clean.columns]

    # Perform aggregation
    df_aggregated = df_clean.groupby(grouping_vars).agg(
        **{var: (var, 'mean') for var in numeric_vars},  # Original names for means
        **{f'{var}_std': (var, 'std') for var in numeric_vars},
        **{f'{var}_count': (var, 'count') for var in numeric_vars},
        n_seeds=('seed', 'nunique')
    ).reset_index()

    print(f"Output dataset shape: {df_aggregated.shape}")

    return df_aggregated

