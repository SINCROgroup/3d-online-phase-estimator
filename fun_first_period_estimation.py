import numpy as np
from scipy.signal import detrend, find_peaks


threshold_acceptable_peaks_wrt_maximum_pcent = 20    # Acceptance range of autocorrelation peaks defined as a percentage of the maximum autocorrelation value.

def compute_signal_period_autocorrelation(pos_signal, vel_signal, local_time_vec, min_length_quasiperiod) -> np.ndarray:
    pos_signal_stacked = np.vstack(pos_signal)
    pos_x_signal = pos_signal_stacked[:, 0]
    pos_y_signal = pos_signal_stacked[:, 1]
    pos_z_signal = pos_signal_stacked[:, 2]

    vel_signal_stacked = np.vstack(vel_signal)
    vel_x_signal = vel_signal_stacked[:, 0]
    vel_y_signal = vel_signal_stacked[:, 1]
    vel_z_signal = vel_signal_stacked[:, 2]

    xn = detrend(pos_x_signal)
    yn = detrend(pos_y_signal)
    zn = detrend(pos_z_signal)
    autocorr_vec_x = compute_autocorr_vec(xn)
    autocorr_vec_y = compute_autocorr_vec(yn)
    autocorr_vec_z = compute_autocorr_vec(zn)
    autocorr_vec_tot = autocorr_vec_x + autocorr_vec_y + autocorr_vec_z

    idx_min_duration = np.argmax(np.array(local_time_vec) > min_length_quasiperiod)
    autocorr_vec_tot[:idx_min_duration] = autocorr_vec_tot[idx_min_duration]

    autocorr_vec_tot = autocorr_vec_tot/np.max(np.abs(autocorr_vec_tot))
 
    peaks, _ = find_peaks(autocorr_vec_tot)
    peaks_values = autocorr_vec_tot[peaks[np.where(peaks > idx_min_duration)]]
    lower_bound_acceptable_peaks_values = max(peaks_values) - (max(peaks_values) * threshold_acceptable_peaks_wrt_maximum_pcent / 100)
    idxs_possible_period = np.where(peaks_values > lower_bound_acceptable_peaks_values)[0]  # indexing necessary because np.where returns a tuple containing an array
    start_idx = 0
    assert len(idxs_possible_period) > 0, "No valid first loop was found, try increasing the listening time."
    end_idx = peaks[idxs_possible_period][0]

    x_period  = pos_x_signal[start_idx:end_idx]
    y_period  = pos_y_signal[start_idx:end_idx]
    z_period  = pos_z_signal[start_idx:end_idx]
    vel_x_period = vel_x_signal[start_idx:end_idx]
    vel_y_period = vel_y_signal[start_idx:end_idx]
    vel_z_period = vel_z_signal[start_idx:end_idx]

    signal_period = np.column_stack((x_period, y_period, z_period, vel_x_period, vel_y_period, vel_z_period))

    return signal_period


def compute_autocorr_vec(signal: np.ndarray) -> np.ndarray:
    max_lag  = len(signal)//2
    autocorr_vec = np.zeros(max_lag + 1)
    
    for lag in range(1, len(autocorr_vec)):
        var = np.var(signal[0:2*lag])
        if var == 0:  autocorr_vec[lag] = 0
        else:
            mean = np.mean(signal[0:2*lag])
            sum_ = 0
            for t in range(lag):
                sum_ += (signal[t] - mean) * (signal[t + lag] - mean)
            autocorr_vec[lag] = sum_ / (lag * var)
    return autocorr_vec
