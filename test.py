from utils.preprocessing import preprocess_all_subjects
import numpy as np
import os

def check_subject(subject_id, data_dir="data/processed"):
    X_path = os.path.join(data_dir, f"X_subject{subject_id}.npy")
    y_path = os.path.join(data_dir, f"y_subject{subject_id}.npy")

    assert os.path.exists(X_path), f"Missing {X_path}"
    assert os.path.exists(y_path), f"Missing {y_path}"

    X = np.load(X_path)
    y = np.load(y_path)

    print(f"✅ Subject {subject_id} | X shape: {X.shape}, y shape: {y.shape}")
    assert X.shape[0] == y.shape[0], "Mismatch: number of trials != number of labels"
    assert len(X.shape) == 3, "X should be (trials, channels, time)"
    assert X.shape[1] == 22, "Expected 22 EEG channels"
    assert X.shape[2] >= 1000, "Expected at least 4s of 250Hz EEG"

def run_test():
    print("📦 Running full preprocessing and validation...")
    preprocess_all_subjects(subject_ids=list(range(1, 10)))

    for sid in range(1, 10):
        check_subject(sid)

if __name__ == "__main__":
    run_test()
