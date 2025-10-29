import pandas as pd
from analysis import clean_data as cd
from analysis import attack_summary as ats


df = pd.read_csv("./results/adversarial_results_MASTER.csv")
df = df.dropna(axis=1, how='all')

# Clean data
df_clean_master = cd.clean_adversarial_data(df)

# Drop irrelevant columns
columns_to_drop = ['eps_z', 'random_start','adv_acc','steps', 'alpha', 'smooth' ,'frac_at_boundary', 'restarts', 'smooth_sigma_t', 'smooth_type', 'targeted', 'top5_roi_share_clean_IG', 'top5_roi_share_adv_IG','top5_roi_share_delta_IG' ]

df_clean_all = df_clean_master.drop(columns=columns_to_drop)

# create attack summary
df, successful_df, threshold = ats.load_and_prepare_data('/Users/temp/Documents/Bath University/Msc/Disseration /EEG_Project/results/adversarial_results_MASTER.csv')
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