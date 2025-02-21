import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from LowPassFilters import LowPassFilter

from OnlineMultidimPhaseEstimator import OnlineMultidimPhaseEstimator


# Parameters
# ------------------------------------------------
discarded_time                 = 0      # [s] all time between start and discarded_time (excluded) is discarded
min_duration_first_quasiperiod = 0      # [s]
listening_time                 = 10     # [s] waits this time before estimating first loop must contain 2 quasiperiods
look_behind_pcent              = 5      # % of last completed loop before last nearest point on which estimate the new phase
look_ahead_pcent               = 15     # % of last completed loop after  last nearest point on which estimate the new phase
time_const_lowpass_filter_phase   = 0.1    # [s]. Use None to disable. Must be larger than time step
# is_use_baseline                = True  # True: tethered mode; False: untethered mode
time_const_lowpass_filter_estimand_pos = 0

file_path_estimand  = r"data\san_giovanni_2024-10-10\spiral_mc_1.csv"
# file_path_estimand  = r"data\san_giovanni_2024-10-10\clockwise_and_anticlockwise_circle_mc_1.csv"; listening_time = 15
# file_path_estimand  = r"data\san_giovanni_2024-10-10\macarena_mc_2.csv"; listening_time = 15
rows_to_skip_estimand = [0, 1, 2] + list(range(4, 40))
col_names_pos_estimand = ['TX.3', 'TY.3', 'TZ.3']
time_step           = 0.01  # [s]
is_use_baseline = True
time_step_baseline  = 0.01
file_path_baseline  = r"data\san_giovanni_2024-10-10\spiral_ref.csv"
col_names_pos_baseline = ['x', 'y', 'z']
col_names_ref_frame_estimand_point_1 = ['TX', 'TY', 'TZ']        # belly
col_names_ref_frame_estimand_point_2 = ['TX.2', 'TY.2', 'TZ.2']  # right chest
col_names_ref_frame_estimand_point_3 = ['TX.1', 'TY.1', 'TZ.1']  # left chest

# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX101_pelvic_balance_good.csv"; col_names_pos_estimand = ["HIP_R_X","HIP_R_Y","HIP_R_Z","ILIAC_R_X","ILIAC_R_Y","ILIAC_R_Z","ILIAC_L_X","ILIAC_L_Y","ILIAC_L_Z","HIP_L_X","HIP_L_Y","HIP_L_Z"];  look_behind_pcent = 5; look_ahead_pcent = 15; listening_time = 15; time_const_lowpass_filter_estimand_pos = 0.1
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX102_pelvic_balance_bad.csv"; col_names_pos_estimand  = ["HIP_R_X","HIP_R_Y","HIP_R_Z","ILIAC_R_X","ILIAC_R_Y","ILIAC_R_Z","ILIAC_L_X","ILIAC_L_Y","ILIAC_L_Z","HIP_L_X","HIP_L_Y","HIP_L_Z"]; look_behind_pcent = 2; look_ahead_pcent = 15; listening_time = 30; time_const_lowpass_filter_estimand_pos = 0.1
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX201_superman_good.csv"; col_names_pos_estimand = ["HAN_R_X","HAN_R_Y","HAN_R_Z", "HAN_L_X", "HAN_L_Y", "HAN_L_Z","ANK_R_X","ANK_R_Y","ANK_R_Z","ANK_L_X","ANK_L_Y","ANK_L_Z"];  look_behind_pcent = 5; look_ahead_pcent = 15; listening_time = 45; time_const_lowpass_filter_estimand_pos = 0.1
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX202_superman_bad.csv"; col_names_pos_estimand  = ["HAN_R_X","HAN_R_Y","HAN_R_Z", "HAN_L_X", "HAN_L_Y", "HAN_L_Z","ANK_R_X","ANK_R_Y","ANK_R_Z","ANK_L_X","ANK_L_Y","ANK_L_Z"];  look_behind_pcent = 5; look_ahead_pcent = 15; listening_time = 25;  time_const_lowpass_filter_estimand_pos = 0.1
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX301_bridge_good.csv"; col_names_pos_estimand = ["HIP_R_X","HIP_R_Y","HIP_R_Z","KNE_R_X","KNE_R_Y","KNE_R_Z","HIP_L_X","HIP_L_Y","HIP_L_Z","KNE_L_X","KNE_L_Y","KNE_L_Z"];  look_behind_pcent = 5; look_ahead_pcent = 30; listening_time = 30; time_const_lowpass_filter_estimand_pos = 0.1
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX302_bridge_bad.csv"; col_names_pos_estimand = ["HIP_R_X","HIP_R_Y","HIP_R_Z","KNE_R_X","KNE_R_Y","KNE_R_Z","HIP_L_X","HIP_L_Y","HIP_L_Z","KNE_L_X","KNE_L_Y","KNE_L_Z"];  look_behind_pcent = 2; look_ahead_pcent = 15; listening_time = 15; time_const_lowpass_filter_estimand_pos = 0.1
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX401_plank_good.csv"; col_names_pos_estimand = ["HEAD_X","HEAD_Y","HEAD_Z","PELVIS_X","PELVIS_Y","PELVIS_Z"];  look_behind_pcent = 5; look_ahead_pcent = 15; listening_time = 22; time_const_lowpass_filter_estimand_pos = 0.1
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX402_plank_bad.csv"; col_names_pos_estimand  = ["HEAD_X","HEAD_Y","HEAD_Z","PELVIS_X","PELVIS_Y","PELVIS_Z"];  discarded_time = 20; look_behind_pcent = 10; look_ahead_pcent = 40; listening_time = 8; time_const_lowpass_filter_estimand_pos = 0.1
# rows_to_skip_estimand = list(range(0, 9))
# time_step = 0.01
# is_use_baseline = False;  time_step_baseline = None


# Load data
# ------------------------------------------------
df_estimand = pd.read_csv(file_path_estimand, skiprows=rows_to_skip_estimand, low_memory=False)
df_estimand_pos = df_estimand[col_names_pos_estimand].copy()
df_estimand_pos.ffill(inplace=True)
estimand_pos_signal = np.array(df_estimand_pos)

time_signal = np.arange(0, time_step * len(df_estimand_pos), time_step)

ref_frame_estimand_points = []
baseline_pos_loop = None
if is_use_baseline:
    df_baseline       = pd.read_csv(file_path_baseline)
    baseline_pos_loop = np.array(df_baseline[col_names_pos_baseline])

    col_names_ref_frame_estimand_points = col_names_ref_frame_estimand_point_1 + col_names_ref_frame_estimand_point_2 + col_names_ref_frame_estimand_point_3
    first_idx_without_na = 0
    for first_idx_without_na in range(len(df_estimand)):
        if df_estimand[col_names_ref_frame_estimand_points].iloc[first_idx_without_na].notna().all():
            break
    ref_frame_estimand_points.append( np.array(df_estimand[col_names_ref_frame_estimand_point_1].iloc[first_idx_without_na]) )
    ref_frame_estimand_points.append( np.array(df_estimand[col_names_ref_frame_estimand_point_2].iloc[first_idx_without_na]) )
    ref_frame_estimand_points.append( np.array(df_estimand[col_names_ref_frame_estimand_point_3].iloc[first_idx_without_na]) )


# Online estimator
# ------------------------------------------------
n_dims_estimand_pos = estimand_pos_signal.shape[1]
n_time_instants     = estimand_pos_signal.shape[0]
phase_estimator = OnlineMultidimPhaseEstimator(
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
    ref_frame_points                = ref_frame_estimand_points
)
phase_estimand_online = np.full(n_time_instants, None)
for i_t in range(n_time_instants - 1):
    phase_estimand_online[i_t] = phase_estimator.update_estimator(estimand_pos_signal[i_t, :], time_signal[i_t])


# Offline estimator
# ------------------------------------------------
# Filter input
if not time_const_lowpass_filter_estimand_pos in {None, 0, -1}:
    estimand_pos_signal_filtered = np.full_like(estimand_pos_signal, np.nan)
    estimand_pos_signal_filtered[0, :] = estimand_pos_signal[0, :]
    low_pass_filter_estimand = LowPassFilter(estimand_pos_signal[0, :], time_signal[1] - time_signal[0], time_const=time_const_lowpass_filter_estimand_pos)
    for i_t in range(1, len(estimand_pos_signal)):
        low_pass_filter_estimand.change_time_step(float(time_signal[i_t] - time_signal[i_t-1]))
        estimand_pos_signal_filtered[i_t, :] = low_pass_filter_estimand.update_state(estimand_pos_signal[i_t, :])
else:
    estimand_pos_signal_filtered = estimand_pos_signal.copy()

means_estimand_pos = np.mean(estimand_pos_signal_filtered, axis=0)  # TODO make this a function or class
estimand_pos_centered = estimand_pos_signal_filtered - means_estimand_pos
pca_model = PCA(n_components=1)
pca_model.fit(estimand_pos_centered)
principal_components = pca_model.transform(estimand_pos_centered)
principal_component_main = principal_components[:, 0]

phase_estimand_offline = np.unwrap(np.angle(hilbert(principal_component_main)))

initial_phase_estimand_online  = phase_estimand_online [int((listening_time + discarded_time) / time_step) + 1]   # + 1 necessary because, e.g., if listening_time + discarded_time = 4, we want to start at time = idx = 5
initial_phase_estimand_offline = phase_estimand_offline[int((listening_time + discarded_time) / time_step) + 1]
phase_estimand_offline += initial_phase_estimand_online - initial_phase_estimand_offline

phase_estimand_offline = np.mod(phase_estimand_offline, 2 * np.pi)

# TODO we should also compute and plot a phase passed through a first order filter (1. unwrap, 2. filter, 3. wrap)


# Figure
# ------------------------------------------------
print(f"Delimiter time instants: {phase_estimator.delimiter_time_instants}")

plt.figure(figsize=(10, 5))
plt.plot(time_signal[int((listening_time + discarded_time) / time_step) + 1:len(phase_estimand_offline) - 900], phase_estimand_offline[int((listening_time + discarded_time) / time_step) + 1:len(phase_estimand_offline) - 900], label='Phase offline')
plt.plot(time_signal[int((listening_time + discarded_time) / time_step) + 1:len(phase_estimand_online) - 900], phase_estimand_online[  int((listening_time + discarded_time) / time_step) + 1:len(phase_estimand_online) - 900],  label='Phase online')
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