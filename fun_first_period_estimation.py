import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import detrend

threshold               = 20    # Acceptance range of autocorrelation peaks defined as a percentage of the maximum autocorrelation value.

def extract_last_period_autocorrelation(x,y,z,vx,vy,vz,t,min_duration_period):
    xn = detrend(x)
    yn = detrend(y)
    zn = detrend(z)
    x_auto = autocor(xn)
    y_auto = autocor(yn)
    z_auto = autocor(zn)
    autocorr = x_auto+y_auto+z_auto
    index_lim = np.argmax(np.array(t) > min_duration_period)
    autocorr[:index_lim]=autocorr[index_lim]
    autocorr=autocorr/np.max(np.abs(autocorr))
 
    peaks, _ = find_peaks(autocorr)
    peaks_values = autocorr[peaks[np.where(peaks > index_lim)]]
    peaks_value_limit = max(peaks_values)-max(peaks_values)*threshold/100
    possible_period = np.where(peaks_values > peaks_value_limit)
    start_idx = 0
    end_inx=peaks[possible_period[0]][0]

    x_last_period = x[start_idx:end_inx]
    y_last_period = y[start_idx:end_inx]
    z_last_period = z[start_idx:end_inx]
    vx_last_period = vx[start_idx:end_inx]
    vy_last_period = vy[start_idx:end_inx]
    vz_last_period = vz[start_idx:end_inx]

    first_period = np.column_stack((x_last_period, y_last_period, z_last_period, vx_last_period, vy_last_period, vz_last_period))

    return first_period


def autocor(signal):   
    max_lag = len(signal)//2
    autocorr = np.zeros(max_lag+1)
    
    for lag in range(1, max_lag + 1):
        sum = 0
        mean = np.mean(signal[0:2*lag])
        var = np.var(signal[0:2*lag])
        if var == 0:
            autocorr[lag] = 0
        else:
            for t in range(lag):
                sum += (signal[t] - mean) * (signal[t + lag] - mean)
            autocorr[lag] = sum / (lag * var)
    return autocorr
