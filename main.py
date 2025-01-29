import numpy as np
import pandas as pd
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from class_phase_estimation import Phase_estimation

## Set parameters
ts                  = 0.01  # step time [s]
wait_time           = 1     # time interval before start phase computation [s]
listening_time      = 10    # max first period interval [s]
min_duration_period = 0     # min first period interval [s] 
percentage_pre      = 0     # % of the last completed period before the last nearest point on which estimate the new phase
percentage_post     = 25    # % of the last completed period after the last nearest point on which estimate the new phase
file_path =r"C:\Users\totos\OneDrive - Università di Napoli Federico II\SHARESPACE\Experiments\experiments_2024-10-10\data\spiral_mc_1.csv"  #path signal
ref_path =r"C:\Users\totos\OneDrive - Università di Napoli Federico II\SHARESPACE\Experiments\experiments_2024-10-10\data\spiral_ref.csv"    #path reference 


ref = pd.read_csv(ref_path)
ref = np.array([ ref['x'],  ref['y'],  ref['z']]).T
Data = pd.read_csv(file_path, skiprows=[0, 1, 2] + list(range(4, 40)),low_memory=False)
T = pd.DataFrame()
T['Var3'] = Data['TX.3']
T['Var4'] = Data['TY.3']
T['Var5'] = Data['TZ.3']
T['Var3'].fillna(method='ffill', inplace=True)
T['Var4'].fillna(method='ffill', inplace=True)
T['Var5'].fillna(method='ffill', inplace=True)
t = np.arange(0, len(T['Var3'])*ts, ts)
trajectory = np.array([T['Var3'], T['Var4'], T['Var5']]).T
for l in range(len(Data)):
    if not pd.isna(Data['TX'][l]) and not pd.isna(Data['TY'][l]) and not pd.isna(Data['TZ'][l]) and \
    not pd.isna(Data['TX.2'][l]) and not pd.isna(Data['TY.2'][l]) and not pd.isna(Data['TZ.2'][l]) and \
    not pd.isna(Data['TX.1'][l]) and not pd.isna(Data['TY.1'][l]) and not pd.isna(Data['TZ.1'][l]):
        break 
belly = np.array([Data['TX'][l], Data['TY'][l], Data['TZ'][l]]) 
right_chest = np.array([Data['TX.2'][l], Data['TY.2'][l], Data['TZ.2'][l]]) 
left_chest = np.array([Data['TX.1'][l], Data['TY.1'][l], Data['TZ.1'][l]])


# online estimator form here
estimator_live=Phase_estimation()
estimator_live.init_setting(ts,percentage_pre,percentage_post,wait_time,listening_time,min_duration_period,ref,left_chest,right_chest,belly)
phase_online = [None] * len(trajectory[:,0])
for j in range(len(trajectory[:,0])-1):
    phase_online[j]=estimator_live.set_position(trajectory[j,:],t[j])

#offline estimtor from here
mean_trajectory = np.mean(trajectory, axis=0)
centered_trajectory = trajectory - mean_trajectory
pca = PCA(n_components=3)
pca.fit(centered_trajectory)
score = pca.transform(centered_trajectory)
principal_component1 = score[:, 0]
phase = np.unwrap(np.angle(hilbert(principal_component1)))
phase += (phase_online[int((listening_time+wait_time)/ts)+1]-phase[int((listening_time+wait_time)/ts)+1])
phase = np.mod(phase, 2*np.pi)

# figure
plt.figure(figsize=(10, 5))
plt.plot(t[int((listening_time+wait_time)/ts)+1:len(phase)-900], phase[int((listening_time+wait_time)/ts)+1:len(phase)-900], label='Phase offline')
plt.plot(t[int((listening_time+wait_time)/ts)+1:len(phase_online)-900], phase_online[int((listening_time+wait_time)/ts)+1:len(phase)-900], label='Phase online')
plt.title('Comparison online-offline estimation', fontsize=16)  
plt.xlabel('Time (s)', fontsize=14)  
plt.ylabel('Phase (radians)', fontsize=14)  
plt.legend(fontsize=12)  
plt.grid(True)
plt.show()