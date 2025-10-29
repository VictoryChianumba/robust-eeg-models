from utils.load_subject import load_subject_data_cached
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_adv_test_data(subject_id):

  train_set, test_set, train_subset, val_subset, adv = load_subject_data_cached("BNCI2014_001", subject_id)
  train_mean_pre, train_std_pre, train_min_pre, train_max_pre = adv

  X_test = np.stack([test_set[i][0] for i in range(len(test_set))])  # (N,C,T
  y_test = np.array(test_set.get_metadata().target)                   # (N,)



  X = torch.tensor(X_test, dtype=torch.float32, device=device)  # (N,C,T)
  y = torch.tensor(y_test, dtype=torch.long, device=device)

  CH_NAMES = train_set.datasets[0].raw.info['ch_names']

  train_min_t = torch.tensor(train_min_pre, dtype=torch.float32, device=device)  # (1,C,1)
  train_max_t = torch.tensor(train_max_pre, dtype=torch.float32, device=device)
  train_std_np = np.array(train_std_pre).reshape(-1)  # (C,)
  train_mean_np = np.array(train_mean_pre).reshape(-1)

  return X, y, train_min_t, train_max_t, train_std_np, train_mean_np, CH_NAMES