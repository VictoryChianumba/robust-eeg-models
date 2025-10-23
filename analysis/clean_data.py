import pandas as pd
import numpy as np
import ast

def clean_adversarial_data(df):
    """
    Clean adversarial results dataset for analysis

    Args:
        df: Raw dataframe from adversarial_results_MASTER.csv

    Returns:
        Cleaned dataframe ready for analysis
    """
    print(f"Initial dataset shape: {df.shape}")

    # 1. Handle Missing Values
    print(f"Missing values in spearman_IG: {df['spearman_IG'].isna().sum()}")
    print(f"Missing values in clean_acc: {df['clean_acc'].isna().sum()}")

    # Drop rows with missing interpretability metrics
    initial_rows = len(df)
    df = df.dropna(subset=['spearman_IG'])
    print(f"Dropped {initial_rows - len(df)} rows with missing interpretability data")

    # 2. Remove Invalid/Incomplete Runs
    # Filter out training failures (suspiciously low clean accuracy)
    training_failures = df['clean_acc'] < 0.3
    print(f"Found {training_failures.sum()} potential training failures (clean_acc < 0.3)")
    df = df[~training_failures]

    # Remove impossible values
    impossible_asr = (df['ASR'] > 1.0) | (df['ASR'] < 0.0)
    impossible_acc = (df['clean_acc'] > 1.0) | (df['clean_acc'] < 0.0) | (df['adv_acc'] > 1.0) | (df['adv_acc'] < 0.0)
    impossible_spearman = (df['spearman_IG'] > 1.0) | (df['spearman_IG'] < -1.0)

    print(f"Impossible ASR values: {impossible_asr.sum()}")
    print(f"Impossible accuracy values: {impossible_acc.sum()}")
    print(f"Impossible Spearman values: {impossible_spearman.sum()}")

    df = df[~(impossible_asr | impossible_acc | impossible_spearman)]

    # 3. Data Type Corrections
    # Convert targeted to proper boolean if it's string
    if df['targeted'].dtype == 'object':
        df['targeted'] = df['targeted'].astype(bool)

    # Handle string representations of lists in eps_uV_per_channel if needed for analysis
    # For now, keep as string since it's not needed for main analysis

    # 4. Consistency Checks
    # Verify ASR = 1 - adv_acc relationship (allowing small floating point errors)
    asr_consistency = np.abs(df['ASR'] - (1 - df['adv_acc'])) > 0.01
    print(f"ASR consistency violations: {asr_consistency.sum()}")
    if asr_consistency.sum() > 0:
        print("Warning: Some ASR values don't match 1 - adv_acc")
        # Fix minor inconsistencies by recalculating ASR
        df['ASR'] = 1 - df['adv_acc']

    # 5. Outlier Detection
    # Check for extreme outliers in SNR that might indicate corrupted attacks
    q1_snr = df['snr_db_mean'].quantile(0.25)
    q3_snr = df['snr_db_mean'].quantile(0.75)
    iqr_snr = q3_snr - q1_snr
    snr_outliers = (df['snr_db_mean'] < (q1_snr - 3*iqr_snr)) | (df['snr_db_mean'] > (q3_snr + 3*iqr_snr))
    print(f"Extreme SNR outliers (3*IQR): {snr_outliers.sum()}")

    # Don't automatically remove SNR outliers - they might be legitimate
    # Just flag them for inspection
    df['snr_outlier_flag'] = snr_outliers

    # 6. Check completeness of experimental design
    # Count unique combinations to verify all experiments completed
    design_check = df.groupby(['model_name', 'subject_id', 'attack', 'muV_budget']).size()
    incomplete_runs = design_check[design_check < 5]  # Expecting 5 seeds
    if len(incomplete_runs) > 0:
        print(f"Found {len(incomplete_runs)} incomplete experimental conditions (< 5 seeds)")
        print("Incomplete runs:")
        print(incomplete_runs.head(10))

    # 7. Add derived variables for analysis
    # Calculate relative accuracy drop
    df['acc_drop'] = df['clean_acc'] - df['adv_acc']
    df['rel_acc_drop'] = df['acc_drop'] / df['clean_acc']

    # Create architecture category
    df['architecture_type'] = df['model_name'].apply(
        lambda x: 'CNN' if x in ['EEGNet', 'DeepConvNet', 'CTNet'] else 'StateSpace'
    )

    # 8. Final quality checks
    print("\nFinal dataset summary:")
    print(f"Final shape: {df.shape}")
    print(f"Models: {df['model_name'].value_counts().to_dict()}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Attack types: {df['attack'].value_counts().to_dict()}")
    print(f"Seeds per condition: {df.groupby(['model_name', 'subject_id', 'attack']).size().describe()}")

    # Check for remaining missing values
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if len(missing_summary) > 0:
        print(f"\nRemaining missing values:")
        print(missing_summary)

    # Summary statistics for key variables
    print(f"\nKey variable ranges:")
    print(f"Clean accuracy: {df['clean_acc'].min():.3f} - {df['clean_acc'].max():.3f}")
    print(f"ASR: {df['ASR'].min():.3f} - {df['ASR'].max():.3f}")
    print(f"Spearman IG: {df['spearman_IG'].min():.3f} - {df['spearman_IG'].max():.3f}")

    return df


