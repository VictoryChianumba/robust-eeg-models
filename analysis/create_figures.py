from analysis import attack_summary as ats
import matplotlib.pyplot as plt

def create_adv_figures(file):


    df, successful_df, threshold = ats.load_and_prepare_data(file)
    summary = ats.calculate_attack_summary(successful_df)

    # Create individual plots
    easr_fig = ats.plot_easr_comparison(summary)
    spearman_fig = ats.plot_spearman_comparison(summary)
    scatter_fig = ats.plot_asr_vs_stability_scatter(summary)
    arch_fig = ats.plot_architecture_comparison(successful_df, threshold)
    detailed_fig = ats.plot_deepfool_detailed_analysis(successful_df)

    # Save plots
    easr_fig.savefig('training/figures/easr_comparison.png', dpi=300, bbox_inches='tight')
    spearman_fig.savefig('training/figures/spearman_comparison.png', dpi=300, bbox_inches='tight')
    scatter_fig.savefig('training/figures/asr_vs_stability.png', dpi=300, bbox_inches='tight')
    arch_fig.savefig('training/figures/architecture_heatmap.png', dpi=300, bbox_inches='tight')
    detailed_fig.savefig('training/figures/deepfool_detailed_analysis.png', dpi=300, bbox_inches='tight')

    plt.show()
