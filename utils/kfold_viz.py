import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def plot_kfold_results(cv_results, save_path=None):
    """
    Comprehensive visualization of k-fold cross-validation results
    
    Args:
        cv_results: Dictionary returned from k_fold_cross_validation()
        save_path: Optional path to save the figure
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('K-Fold Cross-Validation Results', fontsize=16, fontweight='bold')
    
    # Extract data
    fold_accuracies = cv_results['fold_accuracies']
    mean_acc = cv_results['mean_accuracy']
    std_acc = cv_results['std_accuracy']
    
    # 1. Bar plot of fold accuracies
    axes[0, 0].bar(range(1, len(fold_accuracies) + 1), fold_accuracies, 
                   color='skyblue', alpha=0.7, edgecolor='navy')
    axes[0, 0].axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_acc:.3f}')
    axes[0, 0].axhline(y=mean_acc + std_acc, color='orange', linestyle=':', 
                       label=f'±1σ: {std_acc:.3f}')
    axes[0, 0].axhline(y=mean_acc - std_acc, color='orange', linestyle=':')
    axes[0, 0].set_title('Validation Accuracy per Fold', fontweight='bold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, acc in enumerate(fold_accuracies):
        axes[0, 0].text(i+1, acc + 0.005, f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Box plot of accuracies
    axes[0, 1].boxplot(fold_accuracies, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_title('Accuracy Distribution', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_xticklabels(['All Folds'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training curves (if available)
    if 'all_train_losses' in cv_results and cv_results['all_train_losses']:
        for i, losses in enumerate(cv_results['all_train_losses']):
            axes[0, 2].plot(losses, label=f'Fold {i+1}', alpha=0.7)
        axes[0, 2].set_title('Training Loss Curves', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].text(0.5, 0.5, 'Training curves\nnot available', 
                        ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Training Loss Curves', fontweight='bold')
    
    # 4. Validation curves (if available)
    if 'all_val_accuracies' in cv_results and cv_results['all_val_accuracies']:
        for i, accs in enumerate(cv_results['all_val_accuracies']):
            axes[1, 0].plot(accs, label=f'Fold {i+1}', alpha=0.7)
        axes[1, 0].set_title('Validation Accuracy Curves', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Validation curves\nnot available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Validation Accuracy Curves', fontweight='bold')
    
    # 5. Statistical summary
    axes[1, 1].axis('off')
    stats_text = f"""
    Statistical Summary
    
    Mean Accuracy: {mean_acc:.4f}
    Std Deviation: {std_acc:.4f}
    
    Best Fold: {max(fold_accuracies):.4f}
    Worst Fold: {min(fold_accuracies):.4f}
    Range: {max(fold_accuracies) - min(fold_accuracies):.4f}
    
    Coefficient of Variation: {(std_acc/mean_acc)*100:.2f}%
    
    95% Confidence Interval:
    [{mean_acc - 1.96*std_acc:.4f}, {mean_acc + 1.96*std_acc:.4f}]
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    # 6. Performance comparison with random baseline
    random_acc = 0.25  # 25% for 4-class problem
    improvement = ((mean_acc - random_acc) / random_acc) * 100
    
    comparison_data = ['Random\nChance', 'Your Model']
    comparison_values = [random_acc, mean_acc]
    comparison_errors = [0, std_acc]
    
    bars = axes[1, 2].bar(comparison_data, comparison_values, 
                          yerr=comparison_errors, capsize=5,
                          color=['red', 'green'], alpha=0.7)
    axes[1, 2].set_title('Performance vs Random Baseline', fontweight='bold')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add improvement percentage
    axes[1, 2].text(0.5, 0.8, f'+{improvement:.1f}%\nimprovement', 
                    transform=axes[1, 2].transAxes, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Add value labels
    for bar, val, err in zip(bars, comparison_values, comparison_errors):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                        f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def plot_fold_comparison(cv_results):
    """
    Create a detailed comparison between folds
    """
    fold_accuracies = cv_results['fold_accuracies']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Radar chart of fold performance
    angles = np.linspace(0, 2 * np.pi, len(fold_accuracies), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    fold_values = fold_accuracies + [fold_accuracies[0]]  # Complete the circle
    
    axes[0].plot(angles, fold_values, 'o-', linewidth=2, label='Fold Accuracy')
    axes[0].fill(angles, fold_values, alpha=0.25)
    axes[0].set_xticks(angles[:-1])
    axes[0].set_xticklabels([f'Fold {i+1}' for i in range(len(fold_accuracies))])
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Fold Performance Radar Chart', fontweight='bold')
    axes[0].grid(True)
    
    # Statistical test for fold differences
    fold_numbers = list(range(1, len(fold_accuracies) + 1))
    
    # Check if there's significant difference between folds
    if len(fold_accuracies) > 2:
        # One-way ANOVA (though with limited data)
        f_stat, p_value = stats.f_oneway(*[[acc] for acc in fold_accuracies])
        
        axes[1].scatter(fold_numbers, fold_accuracies, s=100, c='blue', alpha=0.7)
        axes[1].plot(fold_numbers, fold_accuracies, 'b--', alpha=0.5)
        
        # Add horizontal line for mean
        mean_line = axes[1].axhline(y=np.mean(fold_accuracies), color='red', 
                                   linestyle='-', linewidth=2, label='Mean')
        
        axes[1].set_xlabel('Fold Number')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Fold-to-Fold Variation', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Add statistical info
        variation_text = f"""
        Fold Variation Analysis
        
        Range: {max(fold_accuracies) - min(fold_accuracies):.4f}
        Std Dev: {np.std(fold_accuracies):.4f}
        CV: {(np.std(fold_accuracies)/np.mean(fold_accuracies))*100:.2f}%
        
        Low variation = stable model
        High variation = unstable model
        """
        
        axes[1].text(0.02, 0.98, variation_text, transform=axes[1].transAxes,
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def print_detailed_results(cv_results):
    """
    Print detailed statistical analysis of k-fold results
    """
    fold_accuracies = cv_results['fold_accuracies']
    mean_acc = cv_results['mean_accuracy']
    std_acc = cv_results['std_accuracy']
    
    print("="*60)
    print("DETAILED K-FOLD CROSS-VALIDATION RESULTS")
    print("="*60)
    
    print(f"\n📊 FOLD-BY-FOLD RESULTS:")
    for i, acc in enumerate(fold_accuracies):
        print(f"   Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"   Best Fold:     {max(fold_accuracies):.4f}")
    print(f"   Worst Fold:    {min(fold_accuracies):.4f}")
    print(f"   Range:         {max(fold_accuracies) - min(fold_accuracies):.4f}")
    
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    cv_percent = (std_acc / mean_acc) * 100
    print(f"   Coefficient of Variation: {cv_percent:.2f}%")
    
    if cv_percent < 5:
        stability = "Very Stable"
    elif cv_percent < 10:
        stability = "Stable"
    elif cv_percent < 20:
        stability = "Moderately Stable"
    else:
        stability = "Unstable"
    
    print(f"   Model Stability: {stability}")
    
    # Confidence interval
    ci_lower = mean_acc - 1.96 * std_acc
    ci_upper = mean_acc + 1.96 * std_acc
    print(f"   95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Comparison with random chance
    random_chance = 0.25  # 4-class problem
    improvement = ((mean_acc - random_chance) / random_chance) * 100
    print(f"\n🚀 BASELINE COMPARISON:")
    print(f"   Random Chance: {random_chance:.4f} (25%)")
    print(f"   Your Model:    {mean_acc:.4f} ({mean_acc*100:.2f}%)")
    print(f"   Improvement:   +{improvement:.1f}%")
    
    if mean_acc > 0.6:
        performance_level = "Excellent"
    elif mean_acc > 0.5:
        performance_level = "Good"
    elif mean_acc > 0.4:
        performance_level = "Fair"
    else:
        performance_level = "Needs Improvement"
    
    print(f"   Performance Level: {performance_level}")
    
    print("="*60)

# Usage example:
# After your k-fold completes:
# plot_kfold_results(cv_results, save_path='kfold_results.png')
# plot_fold_comparison(cv_results)
# print_detailed_results(cv_results)