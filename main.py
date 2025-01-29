import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from class_phase_estimation import Phase_Estimation


# Parameters
# ------------------------------------------------
step_time           = 0.01  # [s]
wait_time           = 1     # time interval before start phase computation [s]
listening_time      = 10    # max first period interval [s]
min_duration_period = 0     # min first period interval [s]
look_behind_pcent   = 0     # % of the last completed period before the last nearest point on which estimate the new phase
look_ahead_pcent    = 25    # % of the last completed period after the last nearest point on which estimate the new phase
file_path_signal    = r"data\spiral_mc_1.csv"
file_path_ref_traj = r"data\spiral_ref.csv"

col_names_traj              = ['TX.3', 'TY.3', 'TZ.3']
col_names_ref_frame_point_1 = ['TX', 'TY', 'TZ']        # belly
col_names_ref_frame_point_2 = ['TX.2', 'TY.2', 'TZ.2']  # right chest
col_names_ref_frame_point_3 = ['TX.1', 'TY.1', 'TZ.1']  # left chest

# Load data
# ------------------------------------------------
df_ref = pd.read_csv(file_path_ref_traj)
df_ref = np.array([df_ref['x'], df_ref['y'], df_ref['z']]).T
data   = pd.read_csv(file_path_signal, skiprows=[0, 1, 2] + list(range(4, 40)), low_memory=False)

df_trajectory = data[col_names_traj].copy()
df_trajectory.ffill(inplace=True)
trajectory = np.array(df_trajectory)
time_vec = np.arange(0, step_time * len(df_trajectory), step_time)

col_names_ref_frame_points = col_names_ref_frame_point_1 + col_names_ref_frame_point_2 + col_names_ref_frame_point_3
first_idx_without_na = 0
for first_idx_without_na in range(len(data)):
    if data[col_names_ref_frame_points].iloc[first_idx_without_na].notna().all():
        break
ref_frame_point_1 = np.array(data[col_names_ref_frame_point_1].iloc[first_idx_without_na])
ref_frame_point_2 = np.array(data[col_names_ref_frame_point_2].iloc[first_idx_without_na])
ref_frame_point_3 = np.array(data[col_names_ref_frame_point_3].iloc[first_idx_without_na])


# Online estimator
# ------------------------------------------------
estimator_live = Phase_Estimation(
    step_time= step_time,
    range_phase_computer_pre  = look_behind_pcent,
    range_phase_computer_post = look_ahead_pcent,
    wait_time                 = wait_time,
    listening_time            = listening_time,
    min_duration_period       = min_duration_period,
    reference                 = df_ref,
    point_1                   = ref_frame_point_3,
    point_2                   = ref_frame_point_2,
    point_3                   = ref_frame_point_1
)
phase_online = [None] * len(trajectory[:, 0])
for j in range(len(trajectory[:,0])-1):
    phase_online[j] = estimator_live.set_position(trajectory[j,:], time_vec[j])


# Offline estimator
# ------------------------------------------------
mean_trajectory = np.mean(trajectory, axis=0)
centered_trajectory = trajectory - mean_trajectory
pca = PCA(n_components=3)
pca.fit(centered_trajectory)
score = pca.transform(centered_trajectory)
principal_component1 = score[:, 0]
phase = np.unwrap(np.angle(hilbert(principal_component1)))
phase += (phase_online[int((listening_time+wait_time) / step_time) + 1] - phase[int((listening_time + wait_time) / step_time) + 1])
phase = np.mod(phase, 2*np.pi)


# Figure
# ------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_vec[int((listening_time + wait_time) / step_time) + 1:len(phase) - 900], phase[int((listening_time + wait_time) / step_time) + 1:len(phase) - 900], label='Phase offline')
plt.plot(time_vec[int((listening_time + wait_time) / step_time) + 1:len(phase_online) - 900], phase_online[int((listening_time + wait_time) / step_time) + 1:len(phase) - 900], label='Phase online')
plt.title('Comparison online-offline estimation', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Phase (radians)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
