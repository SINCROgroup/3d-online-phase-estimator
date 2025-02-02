import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from OnlineMultidimPhaseEstimator import OnlineMultidimPhaseEstimator


# Parameters
# ------------------------------------------------
wait_time           = 1     # time interval before start phase computation [s]
listening_time      = 10    # max first period interval [s]
min_duration_period = 0     # min first period interval [s]
look_behind_pcent   = 0     # % of the last completed period before the last nearest point on which estimate the new phase
look_ahead_pcent    = 25    # % of the last completed period after the last nearest point on which estimate the new phase
file_path_estimand  = r"data\san_giovanni_2024-10-10\spiral_mc_2.csv"
step_time           = 0.01  # [s]
rows_to_skip_estimand = [0, 1, 2] + list(range(4, 40))
col_names_pos_estimand               = ['TX.3', 'TY.3', 'TZ.3']
col_names_ref_frame_estimand_point_1 = ['TX', 'TY', 'TZ']        # belly
col_names_ref_frame_estimand_point_2 = ['TX.2', 'TY.2', 'TZ.2']  # right chest
col_names_ref_frame_estimand_point_3 = ['TX.1', 'TY.1', 'TZ.1']  # left chest


# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX101_pelvic_balance_good.csv"; col_names_pos_estimand = ["HIP_R_X", "HIP_R_Y", "HIP_R_Z"]
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX102_pelvic_balance_bad.csv"; col_names_pos_estimand = ["HIP_R_X", "HIP_R_Y", "HIP_R_Z"]
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX201_superman_good.csv"; col_names_pos_estimand = ["HAN_R_X", "HAN_R_Y", "HAN_R_Z"]
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX202_superman_bad.csv"; col_names_pos_estimand = ["HAN_R_X", "HAN_R_Y", "HAN_R_Z"]
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX301_bridge_good.csv"; col_names_pos_estimand = ["HIP_R_X", "HIP_R_Y", "HIP_R_Z"]
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX302_bridge_bad.csv"; col_names_pos_estimand = ["HIP_R_X", "HIP_R_Y", "HIP_R_Z"]
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX401_plank_good.csv"; col_names_pos_estimand = ["PELVIS_X", "PELVIS_Y", "PELVIS_Z"]
# file_path_estimand = r"data\mocap_exercices_montpellier_2025-01-17\EX402_plank_bad.csv"; col_names_pos_estimand = ["PELVIS_X", "PELVIS_Y", "PELVIS_Z"]
# step_time           = 0.01  # [s]
# listening_time      = 30    # max first period interval [s]

# rows_to_skip_estimand = list(range(0, 9))


# col_names_ref_frame_estimand_point_1 = col_names_pos_estimand
# col_names_ref_frame_estimand_point_2 = col_names_pos_estimand
# col_names_ref_frame_estimand_point_3 = col_names_pos_estimand


file_path_baseline  = r"data\san_giovanni_2024-10-10\spiral_ref.csv"


# Load data
# ------------------------------------------------
df_baseline       = pd.read_csv(file_path_baseline)
baseline_pos_loop = np.array([df_baseline['x'], df_baseline['y'], df_baseline['z']]).T

df_estimand = pd.read_csv(file_path_estimand, skiprows=rows_to_skip_estimand, low_memory=False)
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
phase_estimator = OnlineMultidimPhaseEstimator(
    step_time           = step_time,
    look_behind_pcent   = look_behind_pcent,
    look_ahead_pcent    = look_ahead_pcent,
    wait_time           = wait_time,
    listening_time      = listening_time,
    min_duration_period = min_duration_period,
    baseline_pos_loop   = baseline_pos_loop,
    ref_frame_point_1   = ref_frame_estimand_point_3,
    ref_frame_point_2   = ref_frame_estimand_point_2,
    ref_frame_point_3   = ref_frame_estimand_point_1
)
phase_estimand = np.full(len(estimand_pos_signal[:, 0]), None)
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


# Create a 3D figure
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(estimand_pos_signal[:,0], estimand_pos_signal[:,1], estimand_pos_signal[:,2], label="estimand position signal")
# ax.set_xlabel('X');  ax.set_ylabel('Y');  ax.set_zlabel('Z')
# plt.legend()


plt.show()