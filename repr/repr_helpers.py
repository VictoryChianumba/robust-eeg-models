import random, sys, os, subprocess, torch, numpy as np

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_environment_fingerprint():
    pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
        "pip_freeze": pip_freeze.splitlines(),
    }

def tiny_json(base, model, id, seed, skorch_params, backend, notes):

  os.makedirs(base, exist_ok=True)
  tiny = {
      "model_name": model,
      "subject_id": id,
      "seed": seed,
      "bandpass": {"l_freq": 4.0, "h_freq": 38.0},
      "unit_scale_to_uV": 1e6,
      "ems": {"factor_new": 1e-3, "init_block_size": 750},
      "trial_start_offset_seconds": -0.5,
      "windowing": "create_windows_from_events(session split: 0train/1test)",
      "zscore_applied": False,
      "skorch_params": skorch_params,
      "backend":backend,
      "notes": notes
  }
  return tiny

def safe_model_config(model_config: dict) -> dict:
    """Convert model_config into a JSON-serializable dict."""
    safe_cfg = {}
    for k, v in model_config.items():
        if k == "model_class":
            # store full module path + class name
            safe_cfg[k] = f"{v.__module__}.{v.__name__}" if hasattr(v, "__module__") else str(v)
        elif k == "training":
            safe_training = {}
            for tk, tv in v.items():
                if tk == "optimizer":
                    # store optimizer class name
                    safe_training[tk] = f"{tv.__module__}.{tv.__name__}" if hasattr(tv, "__module__") else str(tv)
                else:
                    safe_training[tk] = tv
            safe_cfg[k] = safe_training
        else:
            safe_cfg[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    return safe_cfg

