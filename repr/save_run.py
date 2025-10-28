from repr import repr_helpers as rp
import torch
import os, json, pickle, numpy as np, numbers

# Define constants
SAVE_DIR = "results"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def _save_run(model_name, subject_id, seed, clf, test_set,
              rng_state_np, rng_state_torch,
              train_idx, val_idx, test_idx,
              train_mean=None, train_std=None,
              train_min=None, train_max=None,
              model_config: dict = None,
              env_fingerprint: dict = None,
              device: str = device):
    """
    clf          : fitted skorch net (clf.module_ is torch.nn.Module)
    test_set     : braindecode Dataset (getitem -> (x, y))
    *_idx        : np.ndarray of ints
    train_mean/std/min/max : arrays shaped (C,) or (1,C,1) (we'll serialize as lists)
    model_config : dict of arch + training hyperparams
    env_fingerprint : dict from get_environment_fingerprint()
    """

    base = f"{SAVE_DIR}/{model_name}/{model_name}_S{subject_id}_seed{seed}"
    os.makedirs(base, exist_ok=True)

    # ---- 0) Metadata header ----
    meta = {
        "model_name": model_name,
        "subject_id": int(subject_id),
        "seed": int(seed),
    }

    if model_config is not None:
        meta["model_config"] = rp.safe_model_config(model_config)
    if env_fingerprint is not None:
        meta["environment"] = env_fingerprint
    json.dump(meta, open(f"{base}/meta.json", "w"), indent=2)

    # ---- 1) Checkpoint (state_dict + optimizer) ----
    torch.save({
        "state_dict": clf.module_.state_dict(),
        "optimizer": getattr(clf, "optimizer_", None).state_dict() if hasattr(clf, "optimizer_") else None,
        "seed": seed
    }, f"{base}/checkpoint.pth")

    # ---- 2) Training curves/history ----
    hist = clf.history_
    # Adjust keys if needed:
    train_acc = [e.get("train_accuracy", e.get("train_acc")) for e in hist]
    val_acc   = [e.get("valid_accuracy", e.get("val_acc")) for e in hist]
    train_loss= [e.get("train_loss") for e in hist]
    val_loss  = [e.get("valid_loss", e.get("val_loss")) for e in hist]
    curves = {"train_acc": train_acc, "val_acc": val_acc,
              "train_loss": train_loss, "val_loss": val_loss}
    json.dump(curves, open(f"{base}/curves.json", "w"), indent=2)

    # ---- 3) Test logits (CLEAN) + loss vector ----
    X_test = np.stack([test_set[i][0] for i in range(len(test_set))])  # (N,C,T)
    y_test = np.array(test_set.get_metadata().target)                  # (N,)
    clf.module_.eval().to(device)
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits_t = clf.infer(X_test_t)   # shape (N, num_classes)
        logits = logits_t.detach().cpu().numpy()
    np.save(f"{base}/test_logits_clean.npy", logits)
    np.save(f"{base}/y_test.npy", y_test)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    y_test_t = torch.tensor(y_test, device=device, dtype=torch.long)
    logits_ten = torch.tensor(logits, device=device, dtype=torch.float32)
    loss_vec = loss_fn(logits_ten, y_test_t).cpu().numpy()
    np.save(f"{base}/test_loss_vector.npy", loss_vec)

    # ---- 4) RNG states ----
    pickle.dump({"numpy": rng_state_np, "torch": rng_state_torch}, open(f"{base}/rng_state.pkl", "wb"))

    # ---- 5) Splits ----
    splits = {"train_idx": train_idx.tolist(),
              "val_idx":   val_idx.tolist(),
              "test_idx":  test_idx.tolist()}
    json.dump(splits, open(f"{base}/splits.json", "w"), indent=2)

    # ---- 6) Preprocessing statistics (per-channel) ----
    prep = {"zscore_applied": False}
    if train_mean is not None:
        prep["train_mean"] = np.array(train_mean).reshape(-1).tolist()
    if train_std is not None:
        prep["train_std"]  = np.array(train_std).reshape(-1).tolist()
    if train_min is not None:
        prep["train_min"]  = np.array(train_min).reshape(-1).tolist()
    if train_max is not None:
        prep["train_max"]  = np.array(train_max).reshape(-1).tolist()
    json.dump(prep, open(f"{base}/preprocessing.json", "w"), indent=2)

    # ---- 7) Attack metadata (placeholder file to append later) ----
    # You will fill this AFTER you run attacks; we create an empty schema now for consistency.
    attack_meta = {
        "whitebox": {},
        "blackbox": {}
    }
    json.dump(attack_meta, open(f"{base}/attack_metadata.json", "w"), indent=2)

    # ---- 8) README for the run folder ----
    with open(f"{base}/README.txt", "w") as f:
        f.write(
            "Artifacts:\n"
            "- checkpoint.pth: model+optimizer state_dict\n"
            "- curves.json: train/val accuracy/loss per epoch\n"
            "- test_logits_clean.npy: logits on test set (clean)\n"
            "- test_loss_vector.npy: per-sample CE loss on test set (clean)\n"
            "- y_test.npy: test labels\n"
            "- rng_state.pkl: RNG snapshots (numpy/torch)\n"
            "- splits.json: train/val/test indices (no leakage)\n"
            "- preprocessing.json: channelwise stats\n"
            "- attack_metadata.json: to be populated after attacks\n"
            "- meta.json: model/subject/seed, environment fingerprint\n"
        )

    # ---------------------------
    # Tiny JSON: per-run manifest
    # ---------------------------
    # grab skorch hyperparams (safe dict)
    try:
        skorch_params = {}
        for k, v in clf.get_params().items():
            if isinstance(v, (str, bool)):
                skorch_params[k] = v
            elif isinstance(v, numbers.Number):
                skorch_params[k] = float(v) if isinstance(v, float) else int(v)
    except Exception:
        skorch_params = {}

    # attach environment + backend determinism fingerprint
    backend = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    # small cache manifest (see function below)

    tiny = rp.tiny_json(
        base, model_name, subject_id, seed, skorch_params, backend,
        notes="baseline training run"
    )

    with open(f"{base}/tiny.json", "w") as f:
        json.dump(tiny, f, indent=2)
