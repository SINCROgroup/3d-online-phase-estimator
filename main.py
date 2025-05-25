import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wrap_functions import wrap_to_2pi
from RecursiveOnlinePhaseEstimator import RecursiveOnlinePhaseEstimator
from compute_phase_via_pca_hilbert import compute_phase_via_pca_hilbert


# Default parameters
# ------------------------------------------------
discarded_time                  = 1      # [s] all time between start and discarded_time (excluded) is discarded
# "listening_time" set below               # [s] waits this time before estimating first loop must contain 2 pseudoperiods
min_duration_first_pseudoperiod = 0      # [s]

look_behind_pcent = 5      # % of last completed loop before last nearest point on which estimate the new phase
look_ahead_pcent  = 15     # % of last completed loop after last nearest point on which estimate the new phase

is_use_baseline = False  # True: tethered mode; False: untethered mode
is_use_elapsed_time = False

time_const_lowpass_filter_phase        = 0.1    # [s]. Use None to disable. Must be larger than time step
time_const_lowpass_filter_estimand_pos = 0.01


# Loaded parameters
#-------------------------------------------------
# Can overwrite default parameters

# from setup_params.san_giovanni_2024_10_10 import *
# from setup_params.montpellier_2025_01_17 import *
# from setup_params.dfki_2025_03_25 import *
# from setup_params.cyens_2025_04_23 import *
from setup_params.cyens_2025_05_28_Ex2 import *
# from setup_params.cyens_2025_05_28_Ex3 import *


# Load data
# ------------------------------------------------
df_estimand = pd.read_csv(file_path_estimand, skiprows=rows_to_skip_estimand, low_memory=False)
df_estimand_pos = df_estimand[col_names_pos_estimand].copy()
# df_estimand_pos = df_estimand[df_estimand.columns[1:]].copy()
df_estimand_pos.ffill(inplace=True)
estimand_pos_signal = np.array(df_estimand_pos)

time_signal = np.arange(0, time_step * len(df_estimand_pos), time_step)

# plt.plot(time_signal, df_estimand_pos); plt.show()

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
    ref_frame_baseline_points = extract_points_from_df(df_baseline, col_names_ref_frame_baseline_points)  # TODO edit point. path B
    # ref_frame_baseline_points = ref_frame_estimand_points.copy()                                            # TODO edit point. path A
else:
    baseline_pos_loop = None
    ref_frame_estimand_points = []
    ref_frame_baseline_points = []
    time_step_baseline = None


# Online estimator
# ------------------------------------------------
n_dims_estimand_pos = estimand_pos_signal.shape[1]
n_time_instants     = estimand_pos_signal.shape[0]
phase_estimator = RecursiveOnlinePhaseEstimator(
    n_dims_estimand_pos             = n_dims_estimand_pos,
    listening_time                  = listening_time,
    discarded_time                  = discarded_time,
    min_duration_first_pseudoperiod = min_duration_first_pseudoperiod,
    look_behind_pcent               = look_behind_pcent,
    look_ahead_pcent                = look_ahead_pcent,
    time_const_lowpass_filter_pos   = time_const_lowpass_filter_estimand_pos,
    time_const_lowpass_filter_phase = time_const_lowpass_filter_phase,
    is_use_baseline                 = is_use_baseline,
    baseline_pos_loop               = baseline_pos_loop,
    time_step_baseline              = time_step_baseline,
    ref_frame_estimand_points       = ref_frame_estimand_points,
    ref_frame_baseline_points       = ref_frame_baseline_points,
    is_use_elapsed_time             = is_use_elapsed_time,
)
phase_estimand_online = np.full(n_time_instants, None)
for i_t in range(n_time_instants - 1):
    phase_estimand_online[i_t] = phase_estimator.update_estimator(estimand_pos_signal[i_t, :], time_signal[i_t])


# Offline estimator
# ------------------------------------------------
phase_estimand_offline = compute_phase_via_pca_hilbert(estimand_pos_signal, time_signal, time_const_lowpass_filter_estimand_pos)

phase_estimand_offline = np.unwrap(phase_estimand_offline)
initial_phase_estimand_online  = phase_estimand_online [int((listening_time + discarded_time) / time_step) + 1]   # + 1 necessary because, e.g., if listening_time + discarded_time = 4, we want to start at time = idx = 5
initial_phase_estimand_offline = phase_estimand_offline[int((listening_time + discarded_time) / time_step) + 1]
phase_estimand_offline += initial_phase_estimand_online - initial_phase_estimand_offline
phase_estimand_offline = wrap_to_2pi(phase_estimand_offline)


# Figure
# ------------------------------------------------
print(f"Delimiter time instants: {phase_estimator.delimiter_time_instants}")

plt.figure(figsize=(10, 5))
plt.plot(time_signal[int((discarded_time) / time_step) + 1:len(phase_estimand_offline)], phase_estimand_offline[int((discarded_time) / time_step) + 1:len(phase_estimand_offline)], label='Phase offline')
plt.plot(time_signal[int((discarded_time) / time_step) + 1:len(phase_estimand_online)], phase_estimand_online[  int((discarded_time) / time_step) + 1:len(phase_estimand_online)],  label='Phase online')
plt.title('Comparison online-offline estimation', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Phase (radians)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)


# Create a 3D figure
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n_loop = 2
# first_idx = np.argmax( time_signal >= phase_estimator.delimiter_time_instants[n_loop-1])
# last_idx  = np.argmax( time_signal > phase_estimator.delimiter_time_instants[n_loop])
# ax.plot(estimand_pos_signal[first_idx:last_idx,0], estimand_pos_signal[first_idx:last_idx,1], estimand_pos_signal[first_idx:last_idx,2], label="estimand position signal")
# ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
# plt.legend()


plt.show()