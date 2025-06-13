import os 
import numpy as np 
import mne 

# constants for filtering and epochs
BANDPASS_LOWER = 8. 
BANDPASS_UPPER = 30
EPOCH_START = 0.0
EPOCH_END = 4.0
# EVENT_ID = dict(left = 769, right = 770, feet = 771, tongue = 772)




def preprocessed_subject(subject_id, data_dir="data/raw", save_dir="data/processed"):
    
    """
    For processing a single subject from BCI IV-2a dataset      
    """ 
    
    filename = f"A0{subject_id}T.gdf"
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist")
    
    raw = mne.io.read_raw_gdf(filepath, preload=True)
    raw.filter(BANDPASS_LOWER, BANDPASS_UPPER, fir_design = 'firwin')
    
    # events, _ = mne.events_from_annotations(raw)
    
    events, event_dict = mne.events_from_annotations(raw)
    print("Event dict:", event_dict)
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id = {
            'left': event_dict['769'],
            'right': event_dict['770'],
            'feet': event_dict['771'],
            'tongue': event_dict['772'],
        },
        tmin = EPOCH_START,
        tmax = EPOCH_END,
        baseline = None, 
        preload = True
    )
    
    X = epochs.get_data()
    y = epochs.events[:, -1] - 769 # from 0 to index    Labels
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"X_subject{subject_id}.npy"), X)
    np.save(os.path.join(save_dir, f"Y_subject{subject_id}.npy"), y)
    
    return X, y

def preprocess_all_subjects(subject_ids=None, data_dir="data/raw", save_dir="data/processed", concatenate=False):
    """

    Args:
        subject_id (_type_, optional): _description_. Defaults to None.
        data_dir (str, optional): _description_. Defaults to "data/raw".
        save_dir (str, optional): _description_. Defaults to "data/processed".
        concatenate (bool, optional): _description_. Defaults to False.
        
    Preprocess multiple subjects and optionally concatenate the data. 
    
    if concatenate=True
    returns: (X_all, Y_all)
    else: None
    """
    
    if subject_ids is None:
        subject_ids = list(range(1, 10))
        
    X_list, y_list = [], []
    
    for subj in subject_ids:
        print(f"preprocessing subject {subj} .....")   
        try:
            X, y = preprocessed_subject(subj, data_dir, save_dir)
            if concatenate:
                X_list.append(X)
                y_list.append(y)
        except Exception as e:
            print(f"[!] Failed on subject {subj}: {e}")
    
    if concatenate:
        X_all= np.concatenate(X_list, axis=0)
        y_all= np.concatenate(y_list, axis=0)
        print(f"Concatenated data shape: X = {X_all.shape}, y = {y_all.shape}")
        return X_all, y_all
    
    print("Preprocessing complete")
    return None
        