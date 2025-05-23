import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wrap_functions import wrap_to_2pi
from OnlineMultidimPhaseEstimator_v2 import OnlineMultidimPhaseEstimator_v2
from compute_phase_via_pca_hilbert import compute_phase_via_pca_hilbert
from Phase_estimator_pca_online import Phase_estimator_pca_online
import csv
import os
import re

# Parameters online multidimentional estimator 
# ------------------------------------------------
discarded_time                 = 5      # [s] all time between start and discarded_time (excluded) is discarded
min_duration_first_quasiperiod = 0      # [s]
listening_time                 = 8    # [s] waits this time before estimating first loop must contain 2 quasiperiods
look_behind_pcent              = 5      # % of last completed loop before last nearest point on which estimate the new phase
look_ahead_pcent               = 15     # % of last completed loop after  last nearest point on which estimate the new phase
time_const_lowpass_filter_phase   = 0   # [s]. Use None to disable. Must be larger than time step
time_const_lowpass_filter_estimand_pos = 0

# Parameters recorded signal 
file_path_estimand  = r"data\san_giovanni_2024-10-10\oscillatingfinger_fdl_1.csv"
rows_to_skip_estimand = [0, 1, 2] + list(range(4, 30))
col_names_pos_estimand = ['TX.3', 'TY.3', 'TZ.3']
time_step           = 0.01  # [s]
is_use_baseline = True
time_step_baseline  = 0.01
directory = os.path.dirname(file_path_estimand)
base_name = os.path.basename(file_path_estimand)
name_moto = re.sub(r'_[a-zA-Z]+_\d+\.csv$', '', base_name)  
file_path_baseline = os.path.join(directory, f"{name_moto}_baseline.csv")
col_names_pos_baseline = ['x', 'y', 'z']
col_names_ref_frame_estimand_points = [['TX', 'TY', 'TZ'], ['TX.2', 'TY.2', 'TZ.2'], ['TX.1', 'TY.1', 'TZ.1']]        # belly, # right chest, # left chest
col_names_ref_frame_baseline_points = [['p1_x','p1_y','p1_z'],['p2_x','p2_y','p2_z'],['p3_x','p3_y','p3_z']]



# Load data
# ------------------------------------------------
df_estimand = pd.read_csv(file_path_estimand, skiprows=rows_to_skip_estimand, low_memory=False)
df_estimand_pos = df_estimand[col_names_pos_estimand].copy()
df_estimand_pos.ffill(inplace=True)
estimand_pos_signal = np.array(df_estimand_pos)

time_signal = np.arange(0, time_step * len(df_estimand_pos), time_step)

def extract_points_from_df(df, col_names_points):
    col_names_points_flattened = []
    for cnp in col_names_points:  col_names_points_flattened.extend(cnp)
    first_idx_without_na = 0
    for first_idx_without_na in range(len(df)):
        if df[col_names_points_flattened].iloc[first_idx_without_na].notna().all():
            break
    points = []
    for i in range(len(col_names_points)):
        points.append(np.array(df[col_names_points[i]].iloc[first_idx_without_na]))
    return points

if is_use_baseline:
    ref_frame_estimand_points = extract_points_from_df(df_estimand, col_names_ref_frame_estimand_points)

    df_baseline       = pd.read_csv(file_path_baseline)
    baseline_pos_loop = np.array(df_baseline[col_names_pos_baseline])
    ref_frame_baseline_points = extract_points_from_df(df_baseline, col_names_ref_frame_baseline_points)                                             # TODO edit point. path A
else:
    baseline_pos_loop = None
    ref_frame_estimand_points = []
    ref_frame_baseline_points = []


# Online estimator multidimensional
# ------------------------------------------------
n_dims_estimand_pos = estimand_pos_signal.shape[1]
n_time_instants     = estimand_pos_signal.shape[0]
phase_estimator = OnlineMultidimPhaseEstimator_v2(
    n_dims_estimand_pos             = n_dims_estimand_pos,
    listening_time                  = listening_time,
    discarded_time                  = discarded_time,
    min_duration_first_quasiperiod  = min_duration_first_quasiperiod,
    look_behind_pcent               = look_behind_pcent,
    look_ahead_pcent                = look_ahead_pcent,
    time_const_lowpass_filter_pos = time_const_lowpass_filter_estimand_pos,
    time_const_lowpass_filter_phase = time_const_lowpass_filter_phase,
    is_use_baseline                 = is_use_baseline,
    baseline_pos_loop               = baseline_pos_loop,
    time_step_baseline              = time_step_baseline,
    ref_frame_estimand_points       = ref_frame_estimand_points,
    ref_frame_baseline_points       = ref_frame_baseline_points,
    is_use_elapsed_time= False,
)
phase_estimand_online = np.full(n_time_instants, None)
for i_t in range(n_time_instants - 1):
    phase_estimand_online[i_t] = phase_estimator.update_estimator(estimand_pos_signal[i_t, :], time_signal[i_t])


# Offline estimator PCA-Hilbert
# ------------------------------------------------
phase_estimand_offline = compute_phase_via_pca_hilbert(estimand_pos_signal, time_signal, time_const_lowpass_filter_estimand_pos)

phase_estimand_offline = np.unwrap(phase_estimand_offline)
initial_phase_estimand_online  = phase_estimand_online [int((listening_time + discarded_time) / time_step) + 1]   # + 1 necessary because, e.g., if listening_time + discarded_time = 4, we want to start at time = idx = 5
initial_phase_estimand_offline = phase_estimand_offline[int((listening_time + discarded_time) / time_step) + 1]
phase_estimand_offline += initial_phase_estimand_online - initial_phase_estimand_offline
phase_estimand_offline = wrap_to_2pi(phase_estimand_offline)



# Real phase
# ------------------------------------------------
file_name_real_phase = file_path_estimand.replace(os.path.basename(file_path_estimand), "real_" + os.path.basename(file_path_estimand))
with open(file_name_real_phase, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    vector_real_period_index = [int(row[0]) for row in reader]

real_phase = np.zeros(vector_real_period_index[-1])

for i in range(len(vector_real_period_index) - 1):
    start_period = vector_real_period_index[i]
    end_period = vector_real_period_index[i + 1]
    real_phase[start_period:end_period] = np.linspace(0, 2 * np.pi, end_period - start_period, endpoint=False)

real_phase = np.unwrap(real_phase)
real_phase += initial_phase_estimand_online - real_phase[int((listening_time + discarded_time) / time_step) + 1]
real_phase = np.mod(real_phase, 2 * np.pi)


# Online estimator PCA_TG
# ------------------------------------------------
window_pca = 3
interval_between_pca = 4
phase_estimator_online_PCA = Phase_estimator_pca_online(window_pca,interval_between_pca)
phase_estimand_online_PCA = np.full(n_time_instants, None)
for i_t in range(n_time_instants - 1):
    phase_estimand_online_PCA[i_t] = phase_estimator_online_PCA.estimate_phase(estimand_pos_signal[i_t, :], time_signal[i_t])
phase_estimand_online_PCA = np.array(phase_estimand_online_PCA, dtype=float)
phase_estimand_online_PCA = np.unwrap(phase_estimand_online_PCA)
initial_phase_estimand_online  = phase_estimand_online [int((listening_time + discarded_time) / time_step) + 1]   # + 1 necessary because, e.g., if listening_time + discarded_time = 4, we want to start at time = idx = 5
initial_phase_estimand_online_PCA = phase_estimand_online_PCA[int((listening_time + discarded_time) / time_step) + 1]
phase_estimand_online_PCA += initial_phase_estimand_online - initial_phase_estimand_online_PCA
phase_estimand_online_PCA = wrap_to_2pi(phase_estimand_online_PCA)

# Figure phase estimation
# ------------------------------------------------
print(f"Delimiter time instants: {phase_estimator.delimiter_time_instants}")
cmap = plt.get_cmap("Dark2")
colors = cmap.colors


first_loop_to_plot = 8
n_loop_to_plot = 2


index_start = np.argmax( time_signal >= phase_estimator.delimiter_time_instants[first_loop_to_plot-1])
plot_window  = np.argmax( time_signal > phase_estimator.delimiter_time_instants[first_loop_to_plot+n_loop_to_plot-1])
plt.figure(figsize=(6,3))
plt.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman'})

plt.plot(time_signal[index_start:plot_window],
         real_phase[index_start:plot_window],
         color=colors[4], linewidth=2,linestyle='--', label='Banchmark')

plt.plot(time_signal[index_start:plot_window],
         phase_estimand_online_PCA[index_start:plot_window],
         color=colors[5], linewidth=2,label='PCA-T')

plt.plot(time_signal[index_start:plot_window],
         phase_estimand_offline[index_start:plot_window],
         color=colors[1], linewidth=2,label='PCA-H')

plt.plot(time_signal[index_start:plot_window],
         phase_estimand_online[index_start:plot_window],
         color=colors[2], linewidth=2,linestyle='-', label='ROPE')

plt.xlabel('Time [s]', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Figure error
# ------------------------------------------------
real_phase = np.array(real_phase, dtype=float)
phase_estimand_online = np.array(phase_estimand_online, dtype=float)
phase_estimand_online_PCA = np.array(phase_estimand_online_PCA, dtype=float)
phase_estimand_offline = np.array(phase_estimand_offline, dtype=float)
error_online = np.abs(np.angle(np.exp(1j * (real_phase[index_start:plot_window] - phase_estimand_online[index_start:plot_window]))))
error_online_PCA = np.abs(np.angle(np.exp(1j * (real_phase[index_start:plot_window] - phase_estimand_online_PCA[index_start:plot_window]))))
error_offline = np.abs(np.angle(np.exp(1j * (real_phase[index_start:plot_window] - phase_estimand_offline[index_start:plot_window]))))
plt.figure(figsize=(6,3))
plt.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman'})

plt.plot(time_signal[index_start:plot_window],
         error_online_PCA,
         color=colors[5], linewidth=2, label='PCA-T')

plt.plot(time_signal[index_start:plot_window],
         error_offline,
         color=colors[1], linewidth=2, label='PCA-H')

plt.plot(time_signal[index_start:plot_window],
         error_online,
         color=colors[2], linewidth=2, linestyle='-', label='ROPE ')


plt.xlabel('Time [s]', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Create a 3D figure
fig = plt.figure()
cmap = plt.get_cmap("Dark2")
colors = cmap.colors
ax = fig.add_subplot(111, projection='3d')
n_loop = 15
first_idx = np.argmax( time_signal >= phase_estimator.delimiter_time_instants[n_loop-1])
last_idx  = np.argmax( time_signal > phase_estimator.delimiter_time_instants[n_loop])
ax.plot(estimand_pos_signal[first_idx:last_idx,0], estimand_pos_signal[first_idx:last_idx,1], estimand_pos_signal[first_idx:last_idx,2], color="C0",linewidth=2)
ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.legend()
plt.show()