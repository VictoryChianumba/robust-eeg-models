import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(filepath):
    """
    Load the adversarial results CSV and prepare data for analysis
    """
    df = pd.read_csv(filepath)

    # Filter for successful attacks only
    successful_attacks = df[df['ASR'] > 0].copy()

    # Calculate median spearman threshold for E-ASR calculation
    median_spearman = successful_attacks['spearman_IG'].median()

    # Calculate E-ASR for each row
    successful_attacks['explanation_fooled'] = successful_attacks['spearman_IG'] < median_spearman

    return df, successful_attacks, median_spearman

def calculate_attack_summary(successful_attacks_df):
    """
    Calculate summary statistics by attack type
    """
    summary = successful_attacks_df.groupby('attack').agg({
        'explanation_fooled': lambda x: (x.sum() / len(x)) * 100,
        'spearman_IG': 'mean',
        'ASR': 'mean',
        'attack': 'count'
    }).round(4)

    summary.columns = ['E_ASR_percent', 'mean_spearman', 'mean_ASR', 'count']
    summary = summary.reset_index()

    return summary

def plot_easr_comparison(summary_df, figsize=(10, 6)):
    """
    Create E-ASR comparison bar plot
    """
    plt.figure(figsize=figsize)


    colors = {
        'FGSM': '#ff6b6b',
        'PGD': '#4ecdc4',
        'FGSM_LP': '#45b7d1',
        'PGD_LP': '#f9ca24',
        'DeepFool_L2': '#6c5ce7'
    }

    attack_colors = [colors.get(attack, '#95a5a6') for attack in summary_df['attack']]

    ax = sns.barplot(data=summary_df, x='attack', y='E_ASR_percent',
                     palette=attack_colors, edgecolor='black', linewidth=0.5)

    plt.title('Explanation Attack Success Rate (E-ASR) by Attack Type', fontsize=14, fontweight='bold')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('E-ASR (%)', fontsize=12)
    plt.ylim(0, 100)

    # Add value labels on bars
    for i, v in enumerate(summary_df['E_ASR_percent']):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_spearman_comparison(summary_df, figsize=(10, 6)):
    """
    Create explanation stability (Spearman) comparison bar plot
    """
    plt.figure(figsize=figsize)

    colors = {
        'FGSM': '#ff6b6b',
        'PGD': '#4ecdc4',
        'FGSM_LP': '#45b7d1',
        'PGD_LP': '#f9ca24',
        'DeepFool_L2': '#6c5ce7'
    }

    attack_colors = [colors.get(attack, '#95a5a6') for attack in summary_df['attack']]

    ax = sns.barplot(data=summary_df, x='attack', y='mean_spearman',
                     palette=attack_colors, edgecolor='black', linewidth=0.5)

    plt.title('Mean Explanation Stability (Spearman Correlation)', fontsize=14, fontweight='bold')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Mean Spearman Correlation', fontsize=12)
    plt.ylim(0.85, 1.0)

    # Add value labels on bars
    for i, v in enumerate(summary_df['mean_spearman']):
        ax.text(i, v , f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def plot_asr_vs_stability_scatter(summary_df, figsize=(10, 8)):
    """
    Create scatter plot of ASR vs Explanation Stability
    """
    plt.figure(figsize=figsize)

    colors = {
        'FGSM': '#ff6b6b',
        'PGD': '#4ecdc4',
        'FGSM_LP': '#45b7d1',
        'PGD_LP': '#f9ca24',
        'DeepFool_L2': '#6c5ce7'
    }

    for _, row in summary_df.iterrows():
        attack = row['attack']
        color = colors.get(attack, '#95a5a6')
        plt.scatter(row['mean_ASR'], row['mean_spearman'],
                   c=color, s=150, alpha=0.8, edgecolors='black', linewidth=1,
                   label=attack)

        # Add attack labels
        plt.annotate(attack, (row['mean_ASR'], row['mean_spearman']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    plt.xlabel('Mean Attack Success Rate', fontsize=12)
    plt.ylabel('Mean Explanation Stability (Spearman)', fontsize=12)
    plt.title('Attack Success Rate vs Explanation Stability', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_architecture_comparison(successful_attacks_df, median_spearman, figsize=(12, 8)):
    """
    Create heatmap of E-ASR by attack type and architecture
    """
    # Calculate E-ASR by attack and architecture
    easr_by_arch = []

    for attack in successful_attacks_df['attack'].unique():
        for model in successful_attacks_df['model_name'].unique():
            subset = successful_attacks_df[
                (successful_attacks_df['attack'] == attack) &
                (successful_attacks_df['model_name'] == model)
            ]

            if len(subset) > 0:
                fooled = (subset['spearman_IG'] < median_spearman).sum()
                easr = (fooled / len(subset)) * 100
                easr_by_arch.append({
                    'attack': attack,
                    'model_name': model,
                    'E_ASR': easr,
                    'count': len(subset)
                })

    easr_df = pd.DataFrame(easr_by_arch)

    # Pivot for heatmap
    heatmap_data = easr_df.pivot(index='model_name', columns='attack', values='E_ASR')

    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r',
                center=50, cbar_kws={'label': 'E-ASR (%)'})
    plt.title('Explanation Attack Success Rate by Architecture and Attack Type',
              fontsize=14, fontweight='bold')
    plt.xlabel('Attack Type', fontsize=12)
    plt.ylabel('Architecture', fontsize=12)
    plt.tight_layout()
    return plt.gcf()

def plot_deepfool_detailed_analysis(successful_attacks_df, figsize=(15, 10)):
    """
    Create comprehensive comparison focusing on DeepFool
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. E-ASR comparison
    summary = calculate_attack_summary(successful_attacks_df)
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']

    axes[0,0].bar(summary['attack'], summary['E_ASR_percent'], color=colors, edgecolor='black')
    axes[0,0].set_title('E-ASR by Attack Type')
    axes[0,0].set_ylabel('E-ASR (%)')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Add value labels
    for i, v in enumerate(summary['E_ASR_percent']):
        axes[0,0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')

    # 2. Spearman comparison
    axes[0,1].bar(summary['attack'], summary['mean_spearman'], color=colors, edgecolor='black')
    axes[0,1].set_title('Mean Explanation Stability')
    axes[0,1].set_ylabel('Spearman Correlation')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].set_ylim(0.85, 1.0)

    # 3. Distribution of Spearman values
    attack_order = ['FGSM', 'PGD', 'FGSM_LP', 'PGD_LP', 'DeepFool_L2']
    sns.boxplot(data=successful_attacks_df, x='attack', y='spearman_IG',
                order=attack_order, ax=axes[1,0])
    axes[1,0].set_title('Distribution of Explanation Stability')
    axes[1,0].set_ylabel('Spearman Correlation')
    axes[1,0].tick_params(axis='x', rotation=45)

    # 4. ASR vs Spearman scatter with DeepFool highlighted
    for attack in summary['attack']:
        subset = successful_attacks_df[successful_attacks_df['attack'] == attack]
        color = colors[list(summary['attack']).index(attack)]
        alpha = 1.0 if attack == 'DeepFool_L2' else 0.6
        size = 60 if attack == 'DeepFool_L2' else 30

        axes[1,1].scatter(subset['ASR'], subset['spearman_IG'],
                         c=color, alpha=alpha, s=size, label=attack)

    axes[1,1].set_xlabel('Attack Success Rate')
    axes[1,1].set_ylabel('Spearman Correlation')
    axes[1,1].set_title('ASR vs Explanation Stability (DeepFool Highlighted)')
    axes[1,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    return fig
