import braindecode
print("Braindecode version:", braindecode.__version__)
from braindecode.models import EEGNetv4, Deep4Net, CTNet
from braindecode.models import CTNet
print("CTNet is available")

from utils.load_adv_data import load_adv_test_data
from repr import repr_helpers as rp
from models.eeg_mamba_fft import EEGMamba
from attack.attack_explainers import AttackExplainers

import torch, os
import pandas as pd
import numpy as np
import captum.attr as CA

SEEDS = [42, 123, 2024, 31415, 999]

SUBJECT_ID = 9  #Subjects 1, 3, 8 & 9 used 

# ---- model builders and checkpoints ----
MODEL_BUILDERS = {
    "EEGNet":      lambda: EEGNetv4(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
    "DeepConvNet": lambda: Deep4Net(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
    "CTNet":       lambda: CTNet(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
    "Mamba":       lambda: EEGMamba(n_chans=n_channels, n_outputs=n_classes, n_times=n_times),
}

CKPTS = {
    "EEGNet":       {s: f"results/EEGNet/EEGNet_S{SUBJECT_ID}_seed{s}/checkpoint.pth"          for s in SEEDS},
    "DeepConvNet":  {s: f"results/DeepConvNet/DeepConvNet_S{SUBJECT_ID}_seed{s}/checkpoint.pth" for s in SEEDS},
    "CTNet":        {s: f"results/CTNet/CTNet_S{SUBJECT_ID}_seed{s}/checkpoint.pth"             for s in SEEDS},
    "Mamba":        {s: f"results/EEGMamba/EEGMamba_S{SUBJECT_ID}_seed{s}/checkpoint.pth"       for s in SEEDS},
}


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

muV_grid = [0.25, 0.5, 1.0, 2.0]  
    
data = load_adv_test_data(SUBJECT_ID)
if isinstance(data, (list, tuple)) and len(data) >= 8:
    X, y, train_min_t, train_max_t, train_std_np, CH_NAMES, train_mean_np, prenorm_std_np = data[:8]
else:
    X, y, train_min_t, train_max_t, train_std_np, train_mean_np, CH_NAMES = data
    # Fallback if prenorm stds not provided by loader (use EMS stds as proxy)
    prenorm_std_np = train_std_np

# map μV -> εᶻ uses PRENORM stats
median_std_pre = float(np.median(prenorm_std_np))

# ROI indices for this subject’s channels
ROI_IDX = [CH_NAMES.index(n) for n in ("C3","C4","Cz") if n in CH_NAMES]


# move data to device
X = X.to(device); y = y.to(device)
train_min_t = train_min_t.to(device); train_max_t = train_max_t.to(device)

ATTR_MAX_N = 128
N_STEPS_IG = 16
IG_INT_BS  = 8
    

def run_for_model_seed(model_name: str, seed: int):

    rp.set_all_seeds(seed)

    # build & load
    m = MODEL_BUILDERS[model_name]().to(device)
    ckpt_path = CKPTS[model_name][seed]
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    m.load_state_dict(state_dict)
    m.eval()
    model = m
    ig = CA.IntegratedGradients(model)

    explainers = AttackExplainers(
        ATTR_MAX_N=ATTR_MAX_N,
        N_STEPS_IG=N_STEPS_IG,
        ig=ig,
        muV_grid=muV_grid,
        model=model,
        X=X,
        y=y,
        train_min_t=train_min_t,
        train_max_t=train_max_t,
        prenorm_std_np=prenorm_std_np,
        median_std_pre=median_std_pre,
        CH_NAMES=None
    )

    # fixed clean subset for attributions
    IDX_CAP = slice(0, min(ATTR_MAX_N, X.size(0)))
    X_cap_clean = X[IDX_CAP].detach().clone()
    y_cap       = y[IDX_CAP].detach().clone()

    # clean explanations cache
    E_CLEAN = {name: fn(X_cap_clean, y_cap) for name, fn in explainers.EXPLAINERS.items()}

    rows = []
    # FGSM / PGD standard
    rows += explainers.run_linf_sweep("FGSM", muV_grid, steps=None, lp_sigma_t=None, cap_idx=IDX_CAP, E_CLEAN=E_CLEAN)
    rows += explainers.run_linf_sweep("PGD",  muV_grid, steps=40,    alpha_rule=lambda e: e/8, lp_sigma_t=None, cap_idx=IDX_CAP, E_CLEAN=E_CLEAN)

    # low-pass (LP) variants (σᵗ=3.0)
    rows += explainers.run_linf_sweep("FGSM", muV_grid, steps=None, lp_sigma_t=3.0, cap_idx=IDX_CAP, E_CLEAN=E_CLEAN)
    rows += explainers.run_linf_sweep("PGD",  muV_grid, steps=40,    alpha_rule=lambda e: e/8, lp_sigma_t=3.0, cap_idx=IDX_CAP, E_CLEAN=E_CLEAN)

    # DeepFool (L2)
    rows += explainers.run_deepfool(cap_idx=IDX_CAP, E_CLEAN=E_CLEAN)

    # tag rows
    for r in rows:
        r["subject_id"] = SUBJECT_ID
        r["model_name"] = model_name
        r["seed"] = seed
    return rows



n_channels = int(X.shape[1])
n_times    = int(X.shape[2])
n_classes  = int(y.max().item() + 1)
print(f"Data shape: {X.shape} | n_ch={n_channels}, n_times={n_times}, n_classes={n_classes}")
print(f"Median PRENORM std (μV): {median_std_pre:.6f} | μV grid: {muV_grid}")

all_rows = []
for model_name in MODEL_BUILDERS.keys():
    print(f"Running: {model_name}")
    for seed in SEEDS:
        all_rows.extend(run_for_model_seed(model_name, seed))

# ---- save per-subject CSV ----
os.makedirs("results", exist_ok=True)
csv_path = f"results/adversarial_results_{SUBJECT_ID}.csv"
pd.DataFrame(all_rows).to_csv(csv_path, index=False)
print(f"Wrote {csv_path} with {len(all_rows)} rows.")

# ---- update master ----
master_path = "results/adversarial_results_MASTER.csv"
if os.path.isfile(master_path):
    pd.concat([pd.read_csv(master_path), pd.DataFrame(all_rows)], ignore_index=True).to_csv(master_path, index=False)
else:
    pd.DataFrame(all_rows).to_csv(master_path, index=False)
print(f"Updated {master_path}.")
