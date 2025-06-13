import os
import mne
from mne.preprocessing import ICA

def load_raw_subject(subject_id, data_dir="data/raw"):
    filepath = os.path.join(f"A0{subject_id}T.gdf")
    raw = mne.io.read_raw_gdf(filepath, preload=True)
    raw.filter(8., 30., fir_design="firwin")
    return raw  

def run_ica_for_inspection(raw, n_components=20, random_state=97):
    ica = ICA(n_components=n_components, random_state=random_state, iter="auto")
    ica.fit(raw)
    return ica

def plot_all_components(ica):
    ica.plot_components()
    
def plot_component_properties(ica, raw, picks=[0, 1, 2]):
    ica.plot_properties(raw, picks)
    
def plot_component_sources(ica, raw):
    ica.plot_sources(raw)