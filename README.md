# Robust EEG Models: Evaluating Adversarial Robustness and Explanation Stability

A systematic evaluation of adversarial robustness in EEG classification models, examining how attack methodology and architecture differences affect both model robustness and the stability of neural network explanations under adversarial perturbation.

## Key Findings

- **Attack methodology is the primary driver of adversarial vulnerability**, not model architecture
- **Explanation fool rate reaches 91% correlation with 50% classification success**, revealing a critical disconnect between classification robustness and explanation stability
- Evaluated three attack methods (FGSM, PGD, DeepFool) across four EEG architectures using physiologically-plausible perturbation budgets (0.25–2.0 μV)

## Quick Start

### Setup
```bash
# Clone repository
git clone https://github.com/VictoryChianumba/robust-eeg-models.git
cd robust-eeg-models

# Create virtual environment and install dependencies
make venv
make install

# Activate environment (after make venv)
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Run Baseline Training
```bash
python main.py
```

### Run Adversarial Attacks
```bash
python adv_run.py
```

### Explore in Jupyter
```bash
jupyter notebook training/train_standard.ipynb
```

## Project Structure
```
robust-eeg-models/
├── main.py                          # Baseline model training & evaluation
├── adv_run.py                       # Adversarial attack pipeline
├── requirements.txt                 # Python dependencies
│
├── training/
│   ├── train_standard.ipynb         # Full pipeline notebook
│   └── figures/                     # Training & analysis visualizations
│
├── models/
│   ├── __init__.py
│   ├── eeg_mamba_fft.py            # EEGMamba architecture (FFT-based)
│   ├── eeg_mamba_moe.py            # EEGMamba variant
│   └── model_note.txt
│
├── attack/
│   ├── attack_explainers.py         
│   └── attack_metrics.py            
│
├── analysis/
│   ├── clean_data.py                # Data cleaning & preprocessing
│   ├── statistical_analysis.py      # Hypothesis testing, effect sizes
│   ├── attack_summary.py            # Aggregate results & reporting
│   └── visualizations.py            # Plotting utilities
│
├── utils/
│   ├── load_adv_data.py             # Load adversarial results
│   ├── load_subject.py              # Load subject-specific EEG data
│   └── train_helpers.py             # Training utilities
│
├── notebooks/
│   ├── 01_preprocessing.ipynb       # EEG preprocessing pipeline
│   ├── 02_training_clean.ipynb      # Model training
│   ├── 03_adversarial_training.ipynb # Adversarial evaluation
│   └── 04_interpretability.ipynb     # Explanation analysis
│
└── repr/                            # [Placeholder: add description]
    ├── repr_helpers.py
    └── save_run.py
```

## Methodology

### Models Evaluated
- **EEGNet**: Compact CNN baseline
- **DeepConvNet**: Deeper convolutional architecture
- **CTNet**: CNN-Transformer hybrid
- **EEGMamba**: CNN-StateSpace hybrid (FFT-based approximation)

### Attack Methods
- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **DeepFool**: Minimum perturbation attack

### Perturbation Budgets
Physiologically-motivated μV-based constraints: `[0.25, 0.5, 1.0, 2.0]`

### Explanation Methods
- Layer-wise Relevance Propagation (LRP)
- Saliency maps
- SHAP values
- Integrated Gradients (IG) with ROI analysis

### Key Metrics
- **Adversarial Success Rate (ASR)**
- **Explanation Stability (Spearman correlation across perturbations)**
- **Explanation Fool Rate** (% explanations that flip while classification remains correct)
- **ROI-level attribution analysis** (region of interest stability)

## Main Results

[INSERT KEY FINDINGS TABLE/SUMMARY]

### Explanation Robustness
- [Spearman correlation results across attacks]
- [Explanation fool rate by model & attack]
- [Disconnect between classification & explanation robustness]

### Attack Effectiveness
- [ASR by attack type]
- [Architecture differences in robustness]
- [Effect size comparisons]

## Data & Reproducibility

**Dataset:** [PLACEHOLDER: describe EEG dataset, number of subjects, sessions, etc.]

**Subjects:** 4 subjects (selected based on data quality)

**Seed Strategy:** Results aggregated across multiple random seeds (see `attack/attack_metrics.py` for statistical controls)

**Statistical Methods:**
- Seed aggregation before analysis (reduced pseudo-replication)
- Fisher's z-transformation for correlation comparisons
- Bootstrap confidence intervals (95% CI)
- Effect sizes: Cohen's d, η²

## Usage Examples

### Train baseline models
```python
from training.train_script import train_model
model = train_model(architecture='eegnet', subject=1)
```

### Run adversarial attacks
```python
from attack.bciadversarialattack import FGSMAttack
attacker = FGSMAttack(model, epsilon=0.5)
x_adv = attacker.attack(x)
```

### Analyze explanation stability
```python
from evaluation.interpretability_metrics import spearman_correlation
correlation = spearman_correlation(explanations_clean, explanations_adv)
```

## Installation & Requirements

Python 3.8+ required.
```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- NumPy, SciPy, Scikit-learn
- [ADD OTHERS FROM requirements.txt]

## Makefile Commands
```bash
make venv        # Create virtual environment
make install     # Install dependencies
make clean       # Remove virtual environment
```

## Citation

[PLACEHOLDER: Add citation for this work once published/available]

## License

[See LICENCE file]

## Contact

[PLACEHOLDER: email or GitHub issues]

---

## Notes

- EEGMamba implementation uses FFT-based approximation, not fully selective state-space architecture
- ROI analysis currently limited to Subject 1 due to computational constraints
- Full IG (Integrated Gradients) collection not completed for all models; see code for status