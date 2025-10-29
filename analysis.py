import pandas as pd
from analysis import clean_data as cd
from analysis.create_figures import create_adv_figures
from analysis.aggregate import aggregate_across_seeds
from analysis import statistical_analysis as sa
from analysis import visualizations as vis

file = "./results/adversarial_results_MASTER.csv"

df = pd.read_csv(file)
df = df.dropna(axis=1, how='all')

# Clean data
df_clean_master = cd.clean_adversarial_data(df)

# Drop irrelevant columns
columns_to_drop = ['eps_z', 'random_start','adv_acc','steps', 'alpha', 'smooth' ,'frac_at_boundary', 'restarts', 'smooth_sigma_t', 'smooth_type', 'targeted', 'top5_roi_share_clean_IG', 'top5_roi_share_adv_IG','top5_roi_share_delta_IG' ]

df_clean_all = df_clean_master.drop(columns=columns_to_drop)

# create attack summary
create_adv_figures(file)


df_aggregated = aggregate_across_seeds(df_clean_all)
df_analysis = sa.statistical_analysis(df_aggregated)

# Create visualizations
fig1, fig2 = vis.create_visualizations(df_analysis)