import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def create_visualizations(df_analysis):
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(20, 16))

    ax1 = plt.subplot(3, 3, 1)

    # Architecture colors
    colors = {'CNN': '#1f77b4', 'StateSpace': '#ff7f0e'}

    for arch in ['CNN', 'StateSpace']:
        subset = df_analysis[df_analysis['architecture_type'] == arch]

        # Scatter plot
        plt.scatter(subset['ASR'], subset['spearman_IG'],
                   alpha=0.6, s=20, color=colors[arch], label=arch)

        # Regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(subset['ASR'], subset['spearman_IG'])
        x_line = np.linspace(subset['ASR'].min(), subset['ASR'].max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color=colors[arch], linestyle='--', alpha=0.8,
                label=f'{arch}: r={r_value:.3f}')

    plt.xlabel('Attack Success Rate (ASR)')
    plt.ylabel('Spearman Correlation (Interpretability Stability)')
    plt.title('Primary Result: ASR vs Interpretability Stability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)

    # Calculate correlations by attack type
    attack_corrs = []
    attack_types = []
    attack_pvals = []

    for attack in df_analysis['attack'].unique():
        subset = df_analysis[df_analysis['attack'] == attack]
        if len(subset) > 10:
            corr, pval = pearsonr(subset['ASR'], subset['spearman_IG'])
            attack_corrs.append(corr)
            attack_types.append(attack)
            attack_pvals.append(pval)

    # Bar plot with significance indicators
    bars = plt.bar(range(len(attack_corrs)), attack_corrs, alpha=0.7)
    plt.xticks(range(len(attack_types)), attack_types, rotation=45)
    plt.ylabel('Correlation (ASR vs Spearman_IG)')
    plt.title('Attack-Specific Correlations')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # Add significance markers
    for i, (bar, pval) in enumerate(zip(bars, attack_pvals)):
        if pval < 0.001:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '***',
                    ha='center', va='bottom', fontsize=12)
        elif pval < 0.01:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '**',
                    ha='center', va='bottom', fontsize=12)
        elif pval < 0.05:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, '*',
                    ha='center', va='bottom', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    ax3 = plt.subplot(3, 3, 3)

    sns.boxplot(data=df_analysis, x='architecture_type', y='ASR', hue='model_name', ax=ax3)
    plt.title('ASR Distribution by Architecture and Model')
    plt.ylabel('Attack Success Rate')
    plt.xlabel('Architecture Type')

    # 4. Perturbation Budget Effects
    ax4 = plt.subplot(3, 3, 4)

    # Filter data with budget information
    df_budget = df_analysis.dropna(subset=['muV_budget'])

    # Plot budget effects by architecture
    for arch in ['CNN', 'StateSpace']:
        subset = df_budget[df_budget['architecture_type'] == arch]
        budget_means = subset.groupby('muV_budget')['ASR'].mean()
        budget_stds = subset.groupby('muV_budget')['ASR'].std()

        plt.errorbar(budget_means.index, budget_means.values, yerr=budget_stds.values,
                    marker='o', label=arch, capsize=5, capthick=2)

    plt.xlabel('Perturbation Budget (μV)')
    plt.ylabel('Mean ASR')
    plt.title('ASR vs Perturbation Budget')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Interpretability Stability by Budget
    ax5 = plt.subplot(3, 3, 5)

    for arch in ['CNN', 'StateSpace']:
        subset = df_budget[df_budget['architecture_type'] == arch]
        budget_means = subset.groupby('muV_budget')['spearman_IG'].mean()
        budget_stds = subset.groupby('muV_budget')['spearman_IG'].std()

        plt.errorbar(budget_means.index, budget_means.values, yerr=budget_stds.values,
                    marker='s', label=arch, capsize=5, capthick=2)

    plt.xlabel('Perturbation Budget (μV)')
    plt.ylabel('Mean Spearman Correlation')
    plt.title('Interpretability Stability vs Budget')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Model-Specific Performance
    ax6 = plt.subplot(3, 3, 6)

    # Clean accuracy vs ASR by model
    model_summary = df_analysis.groupby('model_name').agg({
        'clean_acc': 'mean',
        'ASR': 'mean',
        'spearman_IG': 'mean'
    }).reset_index()

    colors_model = {'EEGNet': '#1f77b4', 'DeepConvNet': '#ff7f0e',
                   'CTNet': '#2ca02c', 'Mamba': '#d62728'}

    for _, row in model_summary.iterrows():
        plt.scatter(row['clean_acc'], row['ASR'], s=100,
                   color=colors_model[row['model_name']],
                   label=row['model_name'], alpha=0.8)

    plt.xlabel('Clean Accuracy')
    plt.ylabel('Mean ASR')
    plt.title('Clean Accuracy vs Adversarial Vulnerability')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. Correlation Matrix Heatmap
    ax7 = plt.subplot(3, 3, 7)

    # Create correlation matrix for key variables
    corr_vars = ['ASR', 'spearman_IG', 'clean_acc', 'snr_db_mean']
    corr_matrix = df_analysis[corr_vars].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                square=True, ax=ax7, cbar_kws={'shrink': 0.8})
    plt.title('Variable Correlation Matrix')

    # 8. Effect Size Visualization
    ax8 = plt.subplot(3, 3, 8)

    # Cohen's d for different comparisons
    effect_sizes = []
    comparisons = []

    # Architecture comparison
    cnn_asr = df_analysis[df_analysis['architecture_type'] == 'CNN']['ASR']
    ss_asr = df_analysis[df_analysis['architecture_type'] == 'StateSpace']['ASR']
    pooled_std = np.sqrt(((len(cnn_asr)-1)*cnn_asr.var() + (len(ss_asr)-1)*ss_asr.var()) /
                        (len(cnn_asr) + len(ss_asr) - 2))
    cohens_d_arch = (cnn_asr.mean() - ss_asr.mean()) / pooled_std
    effect_sizes.append(abs(cohens_d_arch))
    comparisons.append('CNN vs\nStateSpace\n(ASR)')

    # Interpretability comparison
    cnn_ig = df_analysis[df_analysis['architecture_type'] == 'CNN']['spearman_IG']
    ss_ig = df_analysis[df_analysis['architecture_type'] == 'StateSpace']['spearman_IG']
    pooled_std_ig = np.sqrt(((len(cnn_ig)-1)*cnn_ig.var() + (len(ss_ig)-1)*ss_ig.var()) /
                           (len(cnn_ig) + len(ss_ig) - 2))
    cohens_d_ig = (cnn_ig.mean() - ss_ig.mean()) / pooled_std_ig
    effect_sizes.append(abs(cohens_d_ig))
    comparisons.append('CNN vs\nStateSpace\n(Interp.)')

    # Attack type comparison (LP vs standard)
    lp_attacks = df_analysis[df_analysis['attack'].str.contains('_LP', na=False)]['ASR']
    std_attacks = df_analysis[~df_analysis['attack'].str.contains('_LP', na=False) &
                             df_analysis['attack'].isin(['FGSM', 'PGD'])]['ASR']
    if len(lp_attacks) > 0 and len(std_attacks) > 0:
        pooled_std_att = np.sqrt(((len(lp_attacks)-1)*lp_attacks.var() + (len(std_attacks)-1)*std_attacks.var()) /
                                (len(lp_attacks) + len(std_attacks) - 2))
        cohens_d_att = (lp_attacks.mean() - std_attacks.mean()) / pooled_std_att
        effect_sizes.append(abs(cohens_d_att))
        comparisons.append('LP vs\nStandard\n(ASR)')

    bars = plt.bar(range(len(effect_sizes)), effect_sizes, alpha=0.7)
    plt.xticks(range(len(comparisons)), comparisons)
    plt.ylabel("Effect Size (Cohen's d)")
    plt.title('Effect Sizes for Key Comparisons')

    # Add effect size interpretation lines
    plt.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small (0.2)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
    plt.legend(fontsize=8)

    # 9. Residual Analysis
    ax9 = plt.subplot(3, 3, 9)

    # Overall regression residuals
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_analysis['ASR'], df_analysis['spearman_IG'])
    predicted = slope * df_analysis['ASR'] + intercept
    residuals = df_analysis['spearman_IG'] - predicted

    plt.scatter(predicted, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    plt.xlabel('Predicted Spearman_IG')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Overall Regression)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Additional Statistical Summary Figure
    # Create a summary statistics table visualization
    fig2, ax_table = plt.subplots(figsize=(12, 8))
    ax_table.axis('tight')
    ax_table.axis('off')

    # Summary statistics by architecture
    summary_stats = df_analysis.groupby('architecture_type').agg({
        'ASR': ['mean', 'std', 'count'],
        'spearman_IG': ['mean', 'std'],
        'clean_acc': ['mean', 'std']
    }).round(4)

    # Flatten column names
    summary_stats.columns = [f'{col[0]}_{col[1]}' for col in summary_stats.columns]

    # Add correlation values
    arch_correlations = {}
    for arch in ['CNN', 'StateSpace']:
        subset = df_analysis[df_analysis['architecture_type'] == arch]
        corr, pval = pearsonr(subset['ASR'], subset['spearman_IG'])
        arch_correlations[arch] = f"{corr:.4f} (p={pval:.4f})"

    summary_stats['ASR_spearman_correlation'] = [arch_correlations['CNN'],
                                                arch_correlations['StateSpace']]

    # Create table
    table = ax_table.table(cellText=summary_stats.values,
                          rowLabels=summary_stats.index,
                          colLabels=summary_stats.columns,
                          cellLoc='center',
                          loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title('Summary Statistics by Architecture Type', fontsize=16, pad=20)
    plt.show()

    return fig, fig2
