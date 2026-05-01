# When Interpretability Fails: Explanation Blindspots in Adversarial EEG Attacks

Adversarial attacks can cause EEG classifiers to fail while their Integrated Gradients explanations remain nearly unchanged — a silent failure mode for interpretability-based monitoring.

---

## Problem

Interpretability methods are proposed as safety monitors for neural networks. This assumes: if the model fails, the explanation changes.

We test this assumption in EEG motor imagery classification. If an attack can degrade classification while preserving explanations, the monitor is blind to the failure.

---

## Approach

- **Models**: EEGNet, DeepConvNet, CTNet, EEGMamba
- **Dataset**: BCI Competition IV Dataset 2a (4 subjects, 4-class motor imagery)
- **Attacks**: FGSM, PGD, DeepFool-L2, and low-pass variants (FGSM-LP, PGD-LP)
- **Budgets**: 0.25–2.0 μV (physiologically motivated)
- **Explanation method**: Integrated Gradients
- **Stability metric**: Spearman correlation between clean and adversarial attributions

---

## Results

**Baseline accuracy** (chance = 25%):

| Model       | Accuracy |
|-------------|----------|
| CTNet       | 60.6%    |
| EEGNet      | 57.9%    |
| EEGMamba    | 52.7%    |
| DeepConvNet | 49.2%    |

**Adversarial results**:

| Attack type | Mean ASR | Explanation Stability (ρ) |
|-------------|----------|--------------------------|
| Standard    | 93.4%    | 0.872                    |
| Low-pass    | 63.5%    | **0.998**                |

![ASR vs Explanation Stability](results/figures/asr_vs_stability.png)

![Explanation Stability by Attack Type](results/figures/spearman_comparison.png)

![Explanation Attack Success Rate by Architecture](results/figures/architecture_heatmap.png)

Architecture differences were minimal after seed aggregation (Fisher's z comparison: p = 0.71).

---

## Key Insight

**Attack structure matters more than model architecture.**

Standard attacks break both predictions and explanations — a detectable failure. Low-pass attacks break predictions while leaving explanations intact. This is worse: the model is wrong, but the interpretability monitor reports normal.

This reveals a structural vulnerability in using IG explanations as runtime monitors. The explanation and the prediction are not coupled in the way safety arguments require.

---

## Reproduction

```bash
# Setup
git clone https://github.com/VictoryChianumba/robust-eeg-models.git
cd robust-eeg-models
make venv && make install
source venv/bin/activate

# Data: download BCI Competition IV Dataset 2a
# https://www.bbci.de/competition/iv/

# Train baselines
python main.py

# Run adversarial evaluation
python adv_run.py

# Run analysis
python analysis.py

# Blindspot analysis
python run_blindspot.py
```
