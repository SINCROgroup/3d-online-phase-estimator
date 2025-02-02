import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Online3DPhaseEstimator import Online3DPhaseEstimator


# Parameters
# ------------------------------------------------
step_time           = 0.01  # [s]
wait_time           = 1     # time interval before start phase computation [s]
listening_time      = 10    # max first period interval [s]
min_duration_period = 0     # min first period interval [s]
look_behind_pcent   = 0     # % of the last completed period before the last nearest point on which estimate the new phase
look_ahead_pcent    = 25    # % of the last completed period after the last nearest point on which estimate the new phase
file_path_estimand  = r"data\spiral_mc_1.csv"
file_path_baseline  = r"data\spiral_ref.csv"

col_names_pos_estimand               = ['TX.3', 'TY.3', 'TZ.3']
col_names_ref_frame_estimand_point_1 = ['TX', 'TY', 'TZ']        # belly
col_names_ref_frame_estimand_point_2 = ['TX.2', 'TY.2', 'TZ.2']  # right chest
col_names_ref_frame_estimand_point_3 = ['TX.1', 'TY.1', 'TZ.1']  # left chest

# Load data
# ------------------------------------------------
df_baseline       = pd.read_csv(file_path_baseline)
baseline_pos_loop = np.array([df_baseline['x'], df_baseline['y'], df_baseline['z']]).T

df_estimand = pd.read_csv(file_path_estimand, skiprows=[0, 1, 2] + list(range(4, 40)), low_memory=False)  # TODO remove literals
df_estimand_pos = df_estimand[col_names_pos_estimand].copy()
df_estimand_pos.ffill(inplace=True)
estimand_pos_signal = np.array(df_estimand_pos)

time_vec = np.arange(0, step_time * len(df_estimand_pos), step_time)

col_names_ref_frame_estimand_points = col_names_ref_frame_estimand_point_1 + col_names_ref_frame_estimand_point_2 + col_names_ref_frame_estimand_point_3
first_idx_without_na = 0
for first_idx_without_na in range(len(df_estimand)):
    if df_estimand[col_names_ref_frame_estimand_points].iloc[first_idx_without_na].notna().all():
        break
ref_frame_estimand_point_1 = np.array(df_estimand[col_names_ref_frame_estimand_point_1].iloc[first_idx_without_na])
ref_frame_estimand_point_2 = np.array(df_estimand[col_names_ref_frame_estimand_point_2].iloc[first_idx_without_na])
ref_frame_estimand_point_3 = np.array(df_estimand[col_names_ref_frame_estimand_point_3].iloc[first_idx_without_na])


# Online estimator
# ------------------------------------------------
phase_estimator = Online3DPhaseEstimator(
    step_time              = step_time,
    look_behind_pcent      = look_behind_pcent,
    look_ahead_pcent       = look_ahead_pcent,
    wait_time              = wait_time,
    listening_time         = listening_time,
    min_duration_period    = min_duration_period,
    baseline_pos_loop= baseline_pos_loop,
    ref_frame_point_1      = ref_frame_estimand_point_3,
    ref_frame_point_2      = ref_frame_estimand_point_2,
    ref_frame_point_3      = ref_frame_estimand_point_1
)
phase_estimand = [None] * len(estimand_pos_signal[:, 0])
for j in range(len(estimand_pos_signal[:, 0]) - 1):   # TODO rewrite with shape
    phase_estimand[j] = phase_estimator.compute_phase(estimand_pos_signal[j, :], time_vec[j])


# Offline estimator
# ------------------------------------------------
mean_trajectory = np.mean(estimand_pos_signal, axis=0)  # TODO make this a function or class
centered_trajectory = estimand_pos_signal - mean_trajectory
pca = PCA(n_components=3)
pca.fit(centered_trajectory)
score = pca.transform(centered_trajectory)
principal_component1 = score[:, 0]
phase = np.unwrap(np.angle(hilbert(principal_component1)))
phase += (phase_estimand[int((listening_time + wait_time) / step_time) + 1] - phase[int((listening_time + wait_time) / step_time) + 1])
phase = np.mod(phase, 2*np.pi)


# Figure
# ------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_vec[int((listening_time + wait_time) / step_time) + 1:len(phase) - 900], phase[int((listening_time + wait_time) / step_time) + 1:len(phase) - 900], label='Phase offline')
plt.plot(time_vec[int((listening_time + wait_time) / step_time) + 1:len(phase_estimand) - 900], phase_estimand[int((listening_time + wait_time) / step_time) + 1:len(phase) - 900], label='Phase online')
plt.title('Comparison online-offline estimation', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Phase (radians)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
