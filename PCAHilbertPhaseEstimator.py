import numpy as np
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from low_pass_filters import filter_signal

class PCAHilbertPhaseEstimator:
    def __init__(self, estimand_pos_signal, time_signal, time_const_lowpass_filter_estimand_pos=None):
        self.estimand_pos_signal                    = estimand_pos_signal
        self.time_signal                            = time_signal
        self.time_const_lowpass_filter_estimand_pos = time_const_lowpass_filter_estimand_pos


    def compute_phase(self):
        if not self.time_const_lowpass_filter_estimand_pos in {None, 0, -1}:
            self.estimand_pos_signal_filtered = filter_signal(self.estimand_pos_signal, self.time_signal, self.time_const_lowpass_filter_estimand_pos)
        else:
            self.estimand_pos_signal_filtered = self.estimand_pos_signal.copy()

        means_estimand_pos = np.mean(self.estimand_pos_signal_filtered, axis=0)
        estimand_pos_centered = self.estimand_pos_signal_filtered - means_estimand_pos
        pca_model = PCA(n_components=1)
        pca_model.fit(estimand_pos_centered)
        principal_components = pca_model.transform(estimand_pos_centered)
        principal_component_main = principal_components[:, 0]
        return np.angle(hilbert(principal_component_main))