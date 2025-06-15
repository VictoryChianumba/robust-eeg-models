import os 
import numpy as np 
import mne 
from utils.io import get_motor_event_mapping

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
    # make sure the pipeline is not processing the eye movement channels (EOG) and only the 22 EEG channels 
    # These are the 22 EEG channel names used in BCI Competition IV Dataset 2a
    EEG_CHANNELS = ['EEG-Fz', 'EEG-C3', 'EEG-Cz', 'EEG-C4', 'EEG-Pz',
                    'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5',
                    'EEG-6', 'EEG-7', 'EEG-8', 'EEG-9', 'EEG-10',
                    'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-15', 'EEG-16']

    # Retain only valid EEG channels
    raw.pick_channels(EEG_CHANNELS)

    print(f"Selected EEG channels: {len(raw.ch_names)}")
    print(raw.ch_names)  # should be 22 names
    
    assert len(raw.ch_names) == 22, f"Unexpected channel count: {len(raw.ch_names)}"

    raw.filter(BANDPASS_LOWER, BANDPASS_UPPER, fir_design = 'firwin')
    
    # events, _ = mne.events_from_annotations(raw)
    events, event_dict = mne.events_from_annotations(raw)
    event_id, label_map = get_motor_event_mapping(event_dict)

    
    # event_id = {
    #         'left': event_dict['769'],
    #         'right': event_dict['770'],
    #         'feet': event_dict['771'],
    #         'tongue': event_dict['772'],
    #     }
    print("Event dict:", event_dict)
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id = event_id,
        tmin = EPOCH_START,
        tmax = EPOCH_END,
        baseline = None, 
        preload = True
    )
    
    X = epochs.get_data()
    # Get integer class indices 0–3
    # label_map = {
    # event_dict['769']: 0,
    # event_dict['770']: 1,
    # event_dict['771']: 2,
    # event_dict['772']: 3
    # }
    
    # y = np.vectorize(label_map.get)(epochs.events[:, -1])
    
    y_raw = epochs.events[:, -1]
    y = np.vectorize(label_map.get)(y_raw)
    
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
        
        
        



