# ========= Cell 2: Attacks, IG explainer, and sweep runners =========
import numpy as np, torch, torch.nn.functional as F
import torchattacks as ta
from attack import attack_metrics as am

class AttackExplainers():
    def __init__(self, ATTR_MAX_N, N_STEPS_IG, IG_INT_BS, ig):
        self.ATTR_MAX_N = ATTR_MAX_N
        self.N_STEPS_IG = N_STEPS_IG
        self.IG_INT_BS = IG_INT_BS
        self.ig = ig
    
        self.CH_NAMES = None  


def per_channel_clamp(x, vmin, vmax):
    return torch.max(torch.min(x, vmax), vmin)

def snr_db(x, x_adv):
    d = x_adv - x
    num = x.pow(2).sum((1,2)).sqrt()
    den = d.pow(2).sum((1,2)).sqrt().clamp_min(1e-12)
    return (20.0 * torch.log10(num/den)).detach().cpu().numpy()

def per_class_acc(y_true, y_pred, nclass=None):
    yt = y_true.detach().cpu().numpy(); yp = y_pred.detach().cpu().numpy()
    K = int(nclass) if nclass is not None else int(max(yp.max(), yt.max()) + 1)
    out=[]
    for c in range(K):
        idx = (yt == c)
        out.append(float((yp[idx] == c).mean()) if idx.any() else float("nan"))
    return out

def smooth_delta_gauss(delta_t: torch.Tensor, sigma_t: float) -> torch.Tensor:
    # delta_t: (N,C,T). Gaussian LP along time using grouped conv1d (fast, on-device).
    if sigma_t is None or sigma_t <= 0: return delta_t
    device = delta_t.device
    radius = int(4 * sigma_t + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=delta_t.dtype, device=device)
    kernel = torch.exp(-0.5 * (x / sigma_t).pow(2))
    kernel /= kernel.sum()
    C = delta_t.size(1)
    kernel = kernel.expand(C, 1, -1)        # (C,1,K)
    return F.conv1d(delta_t, kernel, padding=radius, groups=C)

# ---------------- μV budgets & mapping (use PRENORM stats) ----------------
# Will be set in setup_subject(): prenorm_std_np, median_std_pre

def eps_from_muV(mu_v):   # scalar ε in z-space
    return float(mu_v / median_std_pre)

def eps_uV_per_channel(eps_z):  # per-channel μV implied by εᶻ
    return (eps_z * prenorm_std_np).tolist()

# ---------------- IG explainer ----------------
def pick_subset(self, X_in, y_in, max_n=None):
    if max_n is None: max_n = self.ATTR_MAX_N
    if max_n and X_in.size(0) > max_n:
        return X_in[:max_n], y_in[:max_n]
    return X_in, y_in

def attr_IG(self, X_in, y_in):
    Xi, yi = pick_subset(X_in, y_in)
    Xi = Xi.detach().clone().requires_grad_(True)
    # internal_batch_size must be >= #examples to avoid Captum warning
    internal_bs = int(Xi.size(0))
    return self.ig.attribute(
        Xi, target=yi, n_steps=self.N_STEPS_IG,
        baselines=torch.zeros_like(Xi),
        internal_batch_size=internal_bs
    )


EXPLAINERS = {"IG": attr_IG}

# ---------------- L_inf sweep (FGSM/PGD) ----------------
def run_linf_sweep(self, atk_name, eps_list_muV, steps=None, alpha_rule=lambda e: e/8,
                   batch=128, seed=42, lp_sigma_t=None, cap_idx=None, E_CLEAN=None):
    global model, X, y, train_min_t, train_max_t, prenorm_std_np

    ctor = {"FGSM": ta.FGSM, "PGD": ta.PGD}[atk_name]
    rows=[]
    N = X.size(0)

    # Prepare subset accumulator
    start = cap_idx.start or 0
    stop  = cap_idx.stop
    cap_N = stop - start
    adv_subset = []

    with torch.no_grad():
        num_classes = int(model(X[:1]).shape[-1])
        preds_clean = model(X).argmax(1).cpu()
        clean_acc = float((preds_clean == y.cpu()).float().mean().item())
        clean_acc_pc = per_class_acc(y.cpu(), preds_clean, num_classes)

    for mu_v in eps_list_muV:
        eps_z = eps_from_muV(mu_v)
        kwargs = {}
        if atk_name == "PGD":
            kwargs["steps"] = steps if steps is not None else 40
            kwargs["alpha"] = float(alpha_rule(eps_z))
            kwargs["random_start"] = True

        atk = ctor(model, eps=float(eps_z), **kwargs)

        preds_adv=[]; snrs=[]; l2_all=[]; l2_succ=[]; linf_all=[]; boundary_hits=[]
        for i in range(0, N, batch):
            Xi = X[i:i+batch].detach().clone().requires_grad_(True)
            yi = y[i:i+batch]
            xa = atk(Xi, yi).detach()

            # Optional LP variant
            if lp_sigma_t is not None:
                delta = xa - Xi
                delta = smooth_delta_gauss(delta, sigma_t=lp_sigma_t)
                delta = delta.clamp(-eps_z, eps_z)  # same L_inf budget in z-space
                xa = Xi + delta

            xa = per_channel_clamp(xa, train_min_t, train_max_t)

            with torch.no_grad():
                pa = model(xa).argmax(1)
            preds_adv.append(pa.cpu())

            d = (xa - Xi).flatten(1)
            l2v = d.norm(p=2, dim=1).detach().cpu().numpy()
            linfv = (xa - Xi).abs().flatten(1).max(dim=1).values.detach().cpu().numpy()
            l2_all.extend(l2v); linf_all.extend(linfv)

            succ = (pa != yi).cpu().numpy()
            l2_succ.extend(l2v[succ])

            snrs.extend(snr_db(Xi, xa))

            # boundary fraction (value-level, not sample-level)
            bmask = ((xa <= (train_min_t + 1e-12)) | (xa >= (train_max_t - 1e-12))).float()
            boundary_hits.append(float(bmask.mean().item()))

            # cache a consistent adversarial subset for explainers
            if len(adv_subset) < cap_N:
                take = min(cap_N - len(adv_subset), xa.size(0))
                adv_subset.append(xa[:take].detach())

        preds_adv = torch.cat(preds_adv)
        X_adv_cap = torch.cat(adv_subset, dim=0)
        y_cap     = y[cap_idx]

        # Explain on clean subset (cached) vs adv subset
        E_ADV = {name: fn(X_adv_cap, y_cap) for name, fn in EXPLAINERS.items()}

        adv_acc = float((preds_adv == y.cpu()).float().mean().item())

        # ---- common row (then add per-method metrics) ----
        row_core = {
            "attack": f"{atk_name}_LP" if lp_sigma_t is not None else atk_name,
            "smooth": bool(lp_sigma_t is not None),
            "smooth_type": "gaussian" if lp_sigma_t is not None else None,
            "smooth_sigma_t": float(lp_sigma_t) if lp_sigma_t is not None else None,
            "norm": "Linf",
            "muV_budget": float(mu_v),
            "eps_z": float(eps_z),
            "eps_uV_per_channel": eps_uV_per_channel(eps_z),
            "steps": kwargs.get("steps"),
            "alpha": float(kwargs["alpha"]) if "alpha" in kwargs else None,
            "random_start": bool(kwargs.get("random_start", False)),
            "seed": seed,
            "clean_acc": clean_acc,
            "adv_acc": adv_acc,
            "ASR": 1.0 - adv_acc,
            "median_L2_success": float(np.median(l2_succ)) if len(l2_succ) else float("nan"),
            "mean_L2_all": float(np.mean(l2_all)) if len(l2_all) else float("nan"),
            "mean_Linf_delta": float(np.mean(linf_all)) if len(linf_all) else float("nan"),
            "snr_db_mean": float(np.mean(snrs)), "snr_db_std": float(np.std(snrs)),
            "clean_acc_per_class": clean_acc_pc,
            "adv_acc_per_class": per_class_acc(y.cpu(), preds_adv, num_classes),
            "frac_at_boundary": float(np.mean(boundary_hits)) if boundary_hits else float("nan"),
            "restarts": 1, "targeted": False
        }

        # ---- per-explainer metrics (IG only) ----
        for m in E_CLEAN.keys():
            Ec, Ea = E_CLEAN[m], E_ADV[m]
            # top-5 lists & ROI share of top-5
            top5_clean = am.topk_channels(Ec, k=5, ch_names=self.CH_NAMES)
            top5_adv   = am.topk_channels(Ea, k=5, ch_names=self.CH_NAMES)
            row_method = {
                f"spearman_{m}":             am.spearman_ch(Ec, Ea),
                f"roi_share_clean_{m}":      am.roi_share(Ec),
                f"roi_share_adv_{m}":        am.roi_share(Ea),
                f"roi_delta_share_{m}":      am.roi_share(Ea) - am.roi_share(Ec),
                f"roi_spearman_{m}":         am.roi_spearman(Ec, Ea),
                f"laterality_clean_{m}":     am.laterality_index(Ec),
                f"laterality_adv_{m}":       am.laterality_index(Ea),
                f"laterality_delta_{m}":     am.laterality_index(Ea) - am.laterality_index(Ec),
                f"top5_clean_{m}":           top5_clean,
                f"top5_adv_{m}":             top5_adv,
                f"top5_roi_share_clean_{m}": am.topk_roi_share(Ec, k=5),
                f"top5_roi_share_adv_{m}":   am.topk_roi_share(Ea, k=5),
                f"top5_roi_share_delta_{m}": am.topk_roi_share(Ea, k=5) - am.topk_roi_share(Ec, k=5),
            }
            rows.append({**row_core, **row_method})

    return rows

# ---------------- DeepFool (L2) ----------------
def run_deepfool(self, batch=128, seed=42, cap_idx=None, E_CLEAN=None):
    atk = ta.DeepFool(model, steps=50)
    preds_adv=[]; snrs=[]; l2_all=[]; l2_succ=[]; linf_all=[]; boundary_hits=[]
    N=X.size(0)

    # subset collector for explanations
    start = cap_idx.start or 0
    stop  = cap_idx.stop
    cap_N = stop - start
    adv_subset = []

    with torch.no_grad():
        preds_clean = model(X).argmax(1).cpu()
        clean_acc = float((preds_clean == y.cpu()).float().mean().item())

    for i in range(0, N, batch):
        Xi = X[i:i+batch].detach().clone().requires_grad_(True)
        yi = y[i:i+batch]
        xa = atk(Xi, yi).detach()
        xa = per_channel_clamp(xa, train_min_t, train_max_t)

        if len(adv_subset) < cap_N:
            take = min(cap_N - len(adv_subset), xa.size(0))
            adv_subset.append(xa[:take].detach())

        with torch.no_grad():
            pa = model(xa).argmax(1)
        preds_adv.append(pa.cpu())

        d = (xa - Xi).flatten(1)
        l2v = d.norm(p=2, dim=1).detach().cpu().numpy()
        l2_all.extend(l2v)
        linfv = (xa - Xi).abs().flatten(1).max(dim=1).values.detach().cpu().numpy()
        linf_all.extend(linfv)

        succ = (pa != yi).cpu().numpy()
        l2_succ.extend(l2v[succ])

        snrs.extend(snr_db(Xi, xa))
        bmask = ((xa <= (train_min_t + 1e-12)) | (xa >= (train_max_t - 1e-12))).float()
        boundary_hits.append(float(bmask.mean().item()))

    preds_adv = torch.cat(preds_adv)
    adv_acc = float((preds_adv == y.cpu()).float().mean().item())

    X_adv_cap = torch.cat(adv_subset, dim=0)
    y_cap     = y[cap_idx]
    E_ADV = {name: fn(X_adv_cap, y_cap) for name, fn in EXPLAINERS.items()}

    rows=[]
    for m in E_CLEAN.keys():
        Ec, Ea = E_CLEAN[m], E_ADV[m]
        top5_clean = am.topk_channels(Ec, k=5, ch_names=self.CH_NAMES)
        top5_adv   = am.topk_channels(Ea, k=5, ch_names=self.CH_NAMES)
        rows.append({
            "attack": "DeepFool_L2",
            "norm": "L2",
            "muV_budget": None,
            "eps_z": None,
            "eps_uV_per_channel": None,
            "steps": 50, "alpha": None, "random_start": None,
            "seed": seed,
            "clean_acc": clean_acc, "adv_acc": adv_acc, "ASR": 1.0 - adv_acc,
            "median_L2_success": float(np.median(l2_succ)) if len(l2_succ) else float("nan"),
            "mean_L2_all": float(np.mean(l2_all)) if len(l2_all) else float("nan"),
            "mean_Linf_delta": float(np.mean(linf_all)) if len(linf_all) else float("nan"),
            "snr_db_mean": float(np.mean(snrs)), "snr_db_std": float(np.std(snrs)),
            "frac_at_boundary": float(np.mean(boundary_hits)) if boundary_hits else float("nan"),
            "restarts": 1, "targeted": False,
            f"spearman_{m}":             am.spearman_ch(Ec, Ea),
            f"roi_share_clean_{m}":      am.roi_share(Ec),
            f"roi_share_adv_{m}":        am.roi_share(Ea),
            f"roi_delta_share_{m}":      am.roi_share(Ea) - am.roi_share(Ec),
            f"roi_spearman_{m}":         am.roi_spearman(Ec, Ea),
            f"laterality_clean_{m}":     am.laterality_index(Ec),
            f"laterality_adv_{m}":       am.laterality_index(Ea),
            f"laterality_delta_{m}":     am.laterality_index(Ea) - am.laterality_index(Ec),
            f"top5_clean_{m}":           top5_clean,
            f"top5_adv_{m}":             top5_adv,
            f"top5_roi_share_clean_{m}": am.topk_roi_share(Ec, k=5),
            f"top5_roi_share_adv_{m}":   am.topk_roi_share(Ea, k=5),
            f"top5_roi_share_delta_{m}": am.topk_roi_share(Ea, k=5) - am.topk_roi_share(Ec, k=5),
        })
    return rows
