import pickle, os, numpy as np
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, exponential_moving_standardize, preprocess
from braindecode.preprocessing import create_windows_from_events
from sklearn.model_selection import train_test_split
from skorch.helper import SliceDataset
from torch.utils.data import Subset

# Global variables:
low_cut_hz = 4.0  # low cut frequency for filtering
high_cut_hz = 38.0  # high cut frequency for filtering

factor_new = 1e-3
init_block_size = 750

def load_subject_data_cached(dataset, subject_id):
    cache_file = f'cache/{dataset}_S{subject_id}_l{low_cut_hz}_h{high_cut_hz}_ems{init_block_size}_{factor_new}.pkl'

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    train_set, test_set, train_subset, val_subset , adv = load_subject_data(dataset,subject_id)

    os.makedirs('cache', exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((train_set, test_set, train_subset, val_subset, adv), f)


    return train_set, test_set, train_subset, val_subset, adv

def load_subject_data(dataset, subject_id):

   

    dataset = MOABBDataset(
        dataset_name=dataset, subject_ids=[subject_id]
    )



    pre_noems = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(lambda data, factor: np.multiply(data, factor), factor=1e6),
        Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz),
    ]
    pre_ems = [
        Preprocessor(exponential_moving_standardize, factor_new=factor_new, init_block_size=init_block_size),
    ]


    preprocess(dataset, pre_noems, n_jobs=-1)



    trial_start_offset_seconds = -0.5

    sfreq = dataset.datasets[0].raw.info["sfreq"]
    assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])

    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    windows_pre = create_windows_from_events(
        dataset,
        trial_start_offset_samples = int(-0.5 * sfreq) ,
        trial_stop_offset_samples = 0,
        preload=True,
        # verbose=0
    )


    # Split into train and test
    splitted = windows_pre.split("session")
    train_set_pre = splitted["0train"]
    test_set_pre = splitted["1test"]

    #Calculate teh train mean, std, min and max
    X_train_pre = np.stack([train_set_pre[i][0] for i in range(len(train_set_pre))])  # (N,C,T)
    train_mean_pre = X_train_pre.mean(axis=(0, 2), keepdims=True)          # (1,C,1)
    train_std_pre  = X_train_pre.std(axis=(0, 2), keepdims=True) + 1e-6    # (1,C,1)
    train_min_pre  = X_train_pre.min(axis=(0, 2), keepdims=True)
    train_max_pre  = X_train_pre.max(axis=(0, 2), keepdims=True)

    # Save prenorm stats + channel names for later mapping/clamping
    ch_names_pre = train_set_pre.datasets[0].raw.info['ch_names']
    os.makedirs("results", exist_ok=True)
    np.savez(f"results/S{subject_id}_prenorm_stats.npz",  # <-- replace 1 with your subject id var
            mean=train_mean_pre, std=train_std_pre,
            vmin=train_min_pre, vmax=train_max_pre,
            ch_names=np.array(ch_names_pre))

    preprocess(dataset, pre_ems, n_jobs=-1)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=int(-0.5 * sfreq),
        trial_stop_offset_samples=0,
        preload=True,
    )

    splitted = windows_dataset.split("session")
    train_set = splitted["0train"]
    test_set  = splitted["1test"]


    # Build simple tensors to compute stats on train windows only
    X_train = SliceDataset(train_set, idx=0)

    y_train = np.array([y for y in SliceDataset(train_set, idx=1)])

    X_test = np.stack([test_set[i][0] for i in range(len(test_set))])  # (N,C,T)
    y_test = np.array(test_set.get_metadata().target)                   # (N,)

    train_indices, val_indices = train_test_split(
          X_train.indices_, test_size=0.2, shuffle=False
      )
    train_subset = Subset(train_set, train_indices)
    val_subset = Subset(train_set, val_indices)
    X_train = np.stack([train_set[i][0] for i in range(len(train_set))])  # (N,C,T

    adv = (train_mean_pre, train_std_pre, train_min_pre, train_max_pre)
    return train_set, test_set, train_subset, val_subset, adv
