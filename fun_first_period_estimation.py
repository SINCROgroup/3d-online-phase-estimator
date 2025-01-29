import numpy as np
from scipy.signal import find_peaks
from scipy.signal import detrend

threshold_acceptable_peaks_wrt_maximum_pcent = 20    # Acceptance range of autocorrelation peaks defined as a percentage of the maximum autocorrelation value.

def extract_last_period_autocorrelation(x_vec, y_vec, z_vec, vel_x_vec, vel_y_vec, vel_z_vec, time_vec, min_duration_period):
    xn = detrend(x_vec)
    yn = detrend(y_vec)
    zn = detrend(z_vec)
    autocorr_vec_x = compute_autocorr_vec(xn)
    autocorr_vec_y = compute_autocorr_vec(yn)
    autocorr_vec_z = compute_autocorr_vec(zn)
    autocorr_vec_tot = autocorr_vec_x + autocorr_vec_y + autocorr_vec_z

    idx_lim = np.argmax(np.array(time_vec) > min_duration_period)
    autocorr_vec_tot[:idx_lim] = autocorr_vec_tot[idx_lim]

    autocorr_vec_tot = autocorr_vec_tot/np.max(np.abs(autocorr_vec_tot))
 
    peaks, _ = find_peaks(autocorr_vec_tot)
    peaks_values = autocorr_vec_tot[peaks[np.where(peaks > idx_lim)]]
    lower_bound_acceptable_peaks_values = max(peaks_values) - (max(peaks_values) * threshold_acceptable_peaks_wrt_maximum_pcent / 100)
    idxs_possible_period = np.where(peaks_values > lower_bound_acceptable_peaks_values)
    start_idx = 0
    end_idx = peaks[idxs_possible_period[0]][0]

    x_last_period  = x_vec[start_idx:end_idx]
    y_last_period  = y_vec[start_idx:end_idx]
    z_last_period  = z_vec[start_idx:end_idx]
    vx_last_period = vel_x_vec[start_idx:end_idx]
    vy_last_period = vel_y_vec[start_idx:end_idx]
    vz_last_period = vel_z_vec[start_idx:end_idx]

    first_period = np.column_stack((x_last_period, y_last_period, z_last_period, vx_last_period, vy_last_period, vz_last_period))

    return first_period


def compute_autocorr_vec(signal):
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
