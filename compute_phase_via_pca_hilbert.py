import numpy as np
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from low_pass_filters import filter_signal

def compute_phase_via_pca_hilbert(estimand_pos_signal, time_signal, time_const_lowpass_filter_estimand_pos=None):
    if not time_const_lowpass_filter_estimand_pos in {None, 0, -1}:
        estimand_pos_signal_filtered = filter_signal(estimand_pos_signal, time_signal, time_const_lowpass_filter_estimand_pos)
    else:
        estimand_pos_signal_filtered = estimand_pos_signal.copy()

    means_estimand_pos = np.mean(estimand_pos_signal_filtered, axis=0)
    estimand_pos_centered = estimand_pos_signal_filtered - means_estimand_pos
    pca_model = PCA(n_components=1)
    pca_model.fit(estimand_pos_centered)
    principal_components = pca_model.transform(estimand_pos_centered)
    principal_component_main = principal_components[:, 0]
    return np.angle(hilbert(principal_component_main))