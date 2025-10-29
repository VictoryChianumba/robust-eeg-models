
from analysis import clean_data as cd
from analysis.aggregate import aggregate_across_seeds
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("./results/adversarial_results_MASTER.csv")
df = df.dropna(axis=1, how='all')

# Clean data
df_clean_master = cd.clean_adversarial_data(df)

# Drop irrelevant columns
columns_to_drop = ['eps_z', 'random_start','adv_acc','steps', 'alpha', 'smooth' ,'frac_at_boundary', 'restarts', 'smooth_sigma_t', 'smooth_type', 'targeted', 'top5_roi_share_clean_IG', 'top5_roi_share_adv_IG','top5_roi_share_delta_IG' ]

df_clean_all = df_clean_master.drop(columns=columns_to_drop)
df_aggregated = aggregate_across_seeds(df_clean_all)

explanation_threshold = 0.001

df = df_aggregated.copy()
df['explanation_fooled'] = (df['spearman_IG'] < explanation_threshold).astype(int)

# Filter the dataset to only include successful attacks
successful_attacks_df = df[df['ASR'] == 1]
explanation_asr = successful_attacks_df['explanation_fooled'].mean()

print(f"Explanation Attack Success Rate (E-ASR): {explanation_asr:.2%}")
print(f"This means {explanation_asr:.2%} of successful attacks also significantly altered the explanation.")


# Plot distribution of spearman_IG
plt.figure(figsize=(8, 5))
sns.histplot(df['spearman_IG'], kde=True, bins=30)
plt.title('Distribution of Spearman IG Values')
plt.xlabel('Spearman IG')
plt.ylabel('Frequency')
plt.show()

# Calculate median spearman_IG for all data and for successful attacks
median_spearman_all = df['spearman_IG'].median()
median_spearman_success = df[df['ASR'] == 1]['spearman_IG'].median()

print(f"Median spearman_IG for all samples: {median_spearman_all:.4f}")
print(f"Median spearman_IG for successful attacks: {median_spearman_success:.4f}")

explanation_threshold = median_spearman_success  

# Create a binary column for explanation fooled
df['explanation_fooled'] = (df['spearman_IG'] < explanation_threshold).astype(int)

# Calculate overall Explanation Attack Success Rate (E-ASR)
successful_attacks_df = df[df['ASR'] == 1]
explanation_asr = successful_attacks_df['explanation_fooled'].mean()
print(f"\nExplanation Attack Success Rate (E-ASR) at threshold {explanation_threshold:.4f}: {explanation_asr:.2%}")

# Group by architecture
e_asr_by_architecture = successful_attacks_df.groupby('architecture_type')['explanation_fooled'].mean()
print("\nE-ASR by Architecture:")
print(e_asr_by_architecture)

# Group by attack type
e_asr_by_attack = successful_attacks_df.groupby('attack')['explanation_fooled'].mean()
print("\nE-ASR by Attack Type:")
print(e_asr_by_attack)

# Group by both architecture and attack
e_asr_by_arch_attack = successful_attacks_df.groupby(['architecture_type', 'attack'])['explanation_fooled'].mean().reset_index()
print("\nE-ASR by Architecture and Attack:")
print(e_asr_by_arch_attack)
plt.figure(figsize=(10, 6))
sns.barplot(data=e_asr_by_arch_attack, x='architecture_type', y='explanation_fooled', hue='attack')
plt.title(f'Explanation Attack Success Rate (E-ASR) at Threshold {explanation_threshold:.2f}')
plt.ylabel('E-ASR (Probability Explanation is Fooled | Attack Successful)')
plt.ylim(0, 1)
plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()