# When Interpretability Fails: Stable Explanations Under Adversarial EEG Attacks

## Summary

We show that Integrated Gradients explanations can remain nearly unchanged under adversarial perturbations that significantly degrade classification accuracy in EEG motor imagery models. This creates an **explanation blindspot**: interpretability signals appear stable even when the model is failing.

---

## Problem

Interpretability methods are often proposed as safety monitors for neural networks.

This assumes:
> if the model fails, the explanation will also change.

We test this assumption in EEG-based BCI systems.

If an adversarial attack can cause misclassification while preserving explanations, then interpretability-based monitoring is unreliable.

---

## Approach

We evaluate:

- **Models**: EEGNet, DeepConvNet, CTNet, EEGMamba  
- **Dataset**: BCI Competition IV 2a (4 subjects)  
- **Attacks**:
  - Standard: FGSM, PGD, DeepFool-L2  
  - Structured: Low-pass variants (FGSM-LP, PGD-LP)  
- **Budgets**: 0.25–2.0 μV (physiologically meaningful)

**Metric:**
- Explanation stability = Spearman correlation between clean and adversarial Integrated Gradients

---

## Results

### Baseline accuracy (4-class, chance = 25%)

| Model | Accuracy |
|------|--------|
| CTNet | 60.6% |
| EEGNet | 57.9% |
| EEGMamba | 52.7% |
| DeepConvNet | 49.2% |

---

### Adversarial performance

| Attack | ASR | Explanation Stability (ρ) |
|--------|-----|--------------------------|
| Standard | 93.4% | 0.872 |
| Low-pass | 63.5% | **0.998** |

---

## Key Result

Low-pass attacks preserve explanations while degrading model performance.

- Classification accuracy drops significantly
- IG explanations remain almost identical (ρ ≈ 0.998)

This creates a **failure mode where the model is wrong but the explanation appears normal**.

---

## Insight

**Attack structure matters more than model architecture.**

- Standard attacks break both predictions and explanations
- Low-pass attacks break predictions while preserving explanations

This is more dangerous: it produces **silent failures** that interpretability cannot detect.

---

## Reproducibility

```bash
pip install -r requirements.txt

# Train baseline
python scripts/train_baselines.py --subject 1 --model eegnet

# Run adversarial attack
python scripts/run_attacks.py --attack fgsm_lp --budget 1.0

# Run analysis
python scripts/run_analysis.py
