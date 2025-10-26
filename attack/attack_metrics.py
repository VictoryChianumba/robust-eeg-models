# ========= Cell 1: Metrics & ROI helpers =========
import numpy as np
import torch
from scipy.stats import spearmanr


def channel_scores(E):  # (N,C,T) -> (N,C)
    return E.abs().mean(dim=-1).detach().cpu().numpy()

def spearman_ch(Ec, Ea):
    sc, sa = channel_scores(Ec), channel_scores(Ea)
    vals=[]
    for i in range(sc.shape[0]):
        r = spearmanr(sc[i], sa[i]).statistic
        if not np.isnan(r): vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")

MOTOR_ROI = ["C3","C4","Cz"]
CH_NAMES = None  

ROI_IDX = [CH_NAMES.index(n) for n in MOTOR_ROI if n in CH_NAMES] if 'CH_NAMES' in globals() and CH_NAMES else None

def roi_share(E, roi_idx=ROI_IDX):
    if roi_idx is None: return float("nan")
    sc = channel_scores(E).mean(axis=0)  # (C,)
    total = sc.sum() + 1e-12
    return float(sc[roi_idx].sum() / total)

def roi_spearman(Ec, Ea, roi_idx=ROI_IDX):
    if roi_idx is None: return float("nan")
    sc, sa = channel_scores(Ec), channel_scores(Ea)
    vals=[]
    for i in range(sc.shape[0]):
        r = spearmanr(sc[i, roi_idx], sa[i, roi_idx]).statistic
        if not np.isnan(r): vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")

def laterality_index(E, left="C3", right="C4"):
    if 'CH_NAMES' not in globals() or not CH_NAMES: return float("nan")
    if left not in CH_NAMES or right not in CH_NAMES: return float("nan")
    li, ri = CH_NAMES.index(left), CH_NAMES.index(right)
    sc = channel_scores(E).mean(axis=0)  # (C,)
    num = sc[li] - sc[ri]
    den = sc[li] + sc[ri] + 1e-8
    return float(num/den)

# ----- Top-k helpers (for report) -----
def _topk_indices_from_E(E, k=5):
    sc_mean = channel_scores(E).mean(axis=0)  # (C,)
    idx = np.argsort(sc_mean)[::-1][:k]
    return idx, sc_mean

def topk_channels(E, k=5, ch_names=CH_NAMES):
    idx, _ = _topk_indices_from_E(E, k=k)
    if ch_names:
        return [ch_names[i] for i in idx]
    return idx.tolist()

def topk_roi_share(E, k=5, roi_idx=ROI_IDX):
    if roi_idx is None: return float("nan")
    idx, _ = _topk_indices_from_E(E, k=k)
    return float(len(set(idx.tolist()).intersection(set(roi_idx))) / max(1,k))
