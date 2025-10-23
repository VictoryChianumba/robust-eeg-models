import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def statistical_analysis(df):
    """
    Comprehensive statistical analysis for interpretability-robustness study
    """
    print("="*60)
    print("STATISTICAL ANALYSIS: INTERPRETABILITY-ROBUSTNESS STUDY")
    print("="*60)

    # Drop redundant columns
    columns_to_drop = ['eps_z', 'random_start','adv_acc','steps', 'alpha', 'smooth' ,
                      'frac_at_boundary', 'restarts', 'smooth_sigma_t', 'smooth_type',
                      'targeted', 'top5_roi_share_clean_IG', 'top5_roi_share_adv_IG',
                      'top5_roi_share_delta_IG']

    df_analysis = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print(f"Analysis dataset shape: {df_analysis.shape}")

    # ================================================================
    # 1. PRIMARY RESEARCH QUESTION: Architecture-specific correlations
    # ================================================================
    print("\n" + "="*50)
    print("1. PRIMARY ANALYSIS: ASR vs Spearman_IG by Architecture")
    print("="*50)

    # Overall correlation
    overall_corr_p = pearsonr(df_analysis['ASR'], df_analysis['spearman_IG'])
    overall_corr_s = spearmanr(df_analysis['ASR'], df_analysis['spearman_IG'])

    print(f"Overall correlation (Pearson): r = {overall_corr_p[0]:.4f}, p = {overall_corr_p[1]:.4f}")
    print(f"Overall correlation (Spearman): rho = {overall_corr_s[0]:.4f}, p = {overall_corr_s[1]:.4f}")

    # Architecture-specific correlations
    print("\nArchitecture-specific correlations:")
    arch_correlations = {}

    for arch in ['CNN', 'StateSpace']:
        subset = df_analysis[df_analysis['architecture_type'] == arch]
        corr_p = pearsonr(subset['ASR'], subset['spearman_IG'])
        corr_s = spearmanr(subset['ASR'], subset['spearman_IG'])

        arch_correlations[arch] = {
            'pearson': corr_p,
            'spearman': corr_s,
            'n': len(subset)
        }

        print(f"{arch}: n={len(subset)}")
        print(f"  Pearson: r = {corr_p[0]:.4f}, p = {corr_p[1]:.4f}")
        print(f"  Spearman: rho = {corr_s[0]:.4f}, p = {corr_s[1]:.4f}")

    # Test difference in correlations using Fisher's z-transform
    def fishers_z_test(r1, n1, r2, n2):
        """Test difference between two correlation coefficients"""
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z2 = 0.5 * np.log((1 + r2) / (1 - r2))
        se = np.sqrt(1/(n1-3) + 1/(n2-3))
        z = (z1 - z2) / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return z, p

    r_cnn = arch_correlations['CNN']['pearson'][0]
    r_ss = arch_correlations['StateSpace']['pearson'][0]
    n_cnn = arch_correlations['CNN']['n']
    n_ss = arch_correlations['StateSpace']['n']

    z_stat, z_p = fishers_z_test(r_cnn, n_cnn, r_ss, n_ss)
    print(f"\nFisher's z-test for correlation difference:")
    print(f"z = {z_stat:.4f}, p = {z_p:.4f}")

    # ================================================================
    # 2. ARCHITECTURE DIFFERENCES IN ROBUSTNESS
    # ================================================================
    print("\n" + "="*50)
    print("2. ARCHITECTURE DIFFERENCES IN ROBUSTNESS")
    print("="*50)

    # ANOVA: ASR ~ architecture + budget + interaction
    # Note: Including budget as categorical for interpretability
    df_analysis['muV_budget_cat'] = pd.Categorical(df_analysis['muV_budget'])

    # Remove rows with missing muV_budget (DeepFool attacks)
    df_budget = df_analysis.dropna(subset=['muV_budget'])

    model_anova = smf.ols('ASR ~ C(architecture_type) * C(muV_budget_cat)', data=df_budget).fit()
    print("ANOVA Results for ASR:")
    print(model_anova.summary().tables[1])

    # Effect sizes (eta-squared)
    ss_total = ((df_budget['ASR'] - df_budget['ASR'].mean())**2).sum()
    ss_model = model_anova.ssr
    eta_squared = ss_model / (ss_model + model_anova.ssr)

    print(f"\nEffect size (eta-squared): {eta_squared:.4f}")

    # Pairwise comparisons between models
    print("\nPairwise model comparisons (Tukey HSD):")
    tukey_result = pairwise_tukeyhsd(df_analysis['ASR'], df_analysis['model_name'])
    print(tukey_result)

    # ================================================================
    # 3. ATTACK-SPECIFIC ANALYSIS
    # ================================================================
    print("\n" + "="*50)
    print("3. ATTACK-SPECIFIC ANALYSIS")
    print("="*50)

    # Correlations by attack type
    print("ASR vs Spearman_IG correlations by attack type:")
    attack_correlations = {}

    for attack in df_analysis['attack'].unique():
        subset = df_analysis[df_analysis['attack'] == attack]
        if len(subset) > 10:  # Sufficient sample size
            corr_p = pearsonr(subset['ASR'], subset['spearman_IG'])
            attack_correlations[attack] = {
                'correlation': corr_p[0],
                'p_value': corr_p[1],
                'n': len(subset)
            }
            print(f"{attack}: r = {corr_p[0]:.4f}, p = {corr_p[1]:.4f}, n = {len(subset)}")

    # ================================================================
    # 4. ROBUST REGRESSION (accounting for subject clustering)
    # ================================================================
    print("\n" + "="*50)
    print("4. ROBUST REGRESSION WITH SUBJECT CLUSTERING")
    print("="*50)

    # Standard regression with clustered standard errors for subject effects
    df_attacks = df_budget[df_budget['attack'].isin(['FGSM', 'PGD', 'FGSM_LP', 'PGD_LP'])]

    model_robust = smf.ols('ASR ~ C(architecture_type) * C(attack)',
                          data=df_attacks).fit(cov_type='cluster',
                                              cov_kwds={'groups': df_attacks['subject_id']})
    print("Robust regression results (clustered by subject):")
    print(model_robust.summary().tables[1])

    # ================================================================
    # 5. INTERACTION EFFECTS: Budget × Architecture
    # ================================================================
    print("\n" + "="*50)
    print("5. INTERACTION EFFECTS: Budget × Architecture")
    print("="*50)

    # Test for interaction in both ASR and spearman_IG
    print("Testing muV_budget × architecture interaction on ASR:")
    interaction_model_asr = smf.ols('ASR ~ muV_budget * C(architecture_type)',
                                   data=df_budget).fit()
    print(f"Interaction coefficient: {interaction_model_asr.params['muV_budget:C(architecture_type)[T.StateSpace]']:.4f}")
    print(f"Interaction p-value: {interaction_model_asr.pvalues['muV_budget:C(architecture_type)[T.StateSpace]']:.4f}")

    print("\nTesting muV_budget × architecture interaction on Spearman_IG:")
    interaction_model_ig = smf.ols('spearman_IG ~ muV_budget * C(architecture_type)',
                                  data=df_budget).fit()
    print(f"Interaction coefficient: {interaction_model_ig.params['muV_budget:C(architecture_type)[T.StateSpace]']:.4f}")
    print(f"Interaction p-value: {interaction_model_ig.pvalues['muV_budget:C(architecture_type)[T.StateSpace]']:.4f}")

    # ================================================================
    # 6. ROBUSTNESS CHECKS
    # ================================================================
    print("\n" + "="*50)
    print("6. ROBUSTNESS CHECKS")
    print("="*50)

    # Bootstrap confidence intervals for main correlation
    def bootstrap_correlation(x, y, n_boot=1000):
        """Bootstrap confidence interval for correlation"""
        n = len(x)
        boot_corrs = []
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            boot_corr = pearsonr(x.iloc[idx], y.iloc[idx])[0]
            boot_corrs.append(boot_corr)
        return np.percentile(boot_corrs, [2.5, 97.5])

    # Overall bootstrap CI
    ci_overall = bootstrap_correlation(df_analysis['ASR'], df_analysis['spearman_IG'])
    print(f"Overall correlation 95% CI: [{ci_overall[0]:.4f}, {ci_overall[1]:.4f}]")

    # Architecture-specific bootstrap CIs
    for arch in ['CNN', 'StateSpace']:
        subset = df_analysis[df_analysis['architecture_type'] == arch]
        ci_arch = bootstrap_correlation(subset['ASR'], subset['spearman_IG'])
        print(f"{arch} correlation 95% CI: [{ci_arch[0]:.4f}, {ci_arch[1]:.4f}]")

    # Outlier sensitivity analysis
    print("\nOutlier sensitivity analysis:")

    # Remove extreme outliers flagged earlier
    if 'snr_outlier_flag' in df_analysis.columns:
        df_no_outliers = df_analysis[~df_analysis['snr_outlier_flag']]
        corr_no_outliers = pearsonr(df_no_outliers['ASR'], df_no_outliers['spearman_IG'])
        print(f"Correlation without SNR outliers: r = {corr_no_outliers[0]:.4f}, p = {corr_no_outliers[1]:.4f}")

    # ================================================================
    # 7. EFFECT SIZES AND PRACTICAL SIGNIFICANCE
    # ================================================================
    print("\n" + "="*50)
    print("7. EFFECT SIZES AND PRACTICAL SIGNIFICANCE")
    print("="*50)

    # Cohen's d for architecture differences
    cnn_asr = df_analysis[df_analysis['architecture_type'] == 'CNN']['ASR']
    ss_asr = df_analysis[df_analysis['architecture_type'] == 'StateSpace']['ASR']

    pooled_std = np.sqrt(((len(cnn_asr)-1)*cnn_asr.var() + (len(ss_asr)-1)*ss_asr.var()) /
                        (len(cnn_asr) + len(ss_asr) - 2))
    cohens_d = (cnn_asr.mean() - ss_asr.mean()) / pooled_std

    print(f"Cohen's d for architecture difference in ASR: {cohens_d:.4f}")

    # Interpretation of correlation magnitudes
    r_overall = overall_corr_p[0]
    if abs(r_overall) < 0.1:
        magnitude = "negligible"
    elif abs(r_overall) < 0.3:
        magnitude = "small"
    elif abs(r_overall) < 0.5:
        magnitude = "medium"
    else:
        magnitude = "large"

    print(f"Overall correlation magnitude: {magnitude} (r = {r_overall:.4f})")

    # Summary statistics table
    print("\n" + "="*50)
    print("8. SUMMARY STATISTICS BY ARCHITECTURE")
    print("="*50)

    summary_stats = df_analysis.groupby('architecture_type')[['ASR', 'spearman_IG', 'clean_acc']].agg(['mean', 'std', 'count'])
    print(summary_stats.round(4))

    return df_analysis
