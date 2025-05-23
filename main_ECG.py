import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
from wrap_functions import wrap_to_2pi
from RecursiveOnlinePhaseEstimator import RecursiveOnlinePhaseEstimator
from compute_phase_via_pca_hilbert import compute_phase_via_pca_hilbert
from Phase_estimator_pca_online import Phase_estimator_pca_online
import os
import csv


#number_iterations=['4']
number_iterations=['4']
shapes=['ECG']
shapes_name=['ECG']
file_path_base = 'data\\ECG'
base_colors = ["C0", "C1", "C2", "C4"]

# Parameters
# ------------------------------------------------
discarded_time                 = 1    # [s] all time between start and discarded_time (excluded) is discarded
min_duration_first_quasiperiod = 0.6      # [s]
listening_time                 = 4   # [s] waits this time before estimating first loop must contain 2 quasiperiods
look_behind_pcent              = 5     # % of last completed loop before last nearest point on which estimate the new phase
look_ahead_pcent               = 15  # % of last completed loop after last nearest point on which estimate the new phase
time_const_lowpass_filter_phase   = 0.01    # [s]. Use None to disable. Must be larger than time step
is_use_baseline                = False  # True: tethered mode; False: untethered mode
time_const_lowpass_filter_estimand_pos = 0
baseline_pos_loop = None
ref_frame_estimand_points = []
ref_frame_baseline_points = []
error_dict = {shape: {'ROPE': [], 'PCA-T': [], 'PCA-H': []} for shape in shapes}


for i, shape in enumerate(shapes):
    all_errors_online = []
    all_errors_online_PCA = []
    all_errors_offline = []
    for iteration in number_iterations:
        file_path_estimand = f"{file_path_base}\\{shape}_{iteration}.csv"
        df_ecg = pd.read_csv(file_path_estimand, header=None)
        df_ecg.columns = ['Time', 'Channel1', 'Channel2']
        estimand_pos_signal = df_ecg['Channel1'].iloc[:30*360].to_numpy().reshape(-1, 1)
        time_step           = 1/360  # [s]
        is_use_baseline = False
        time_step_baseline  = 0.01
        time_signal = np.arange(0, time_step * len(estimand_pos_signal), time_step)

        # Online estimator
        # ------------------------------------------------
        n_dims_estimand_pos = estimand_pos_signal.shape[1]
        n_time_instants     = estimand_pos_signal.shape[0]
        phase_estimator = RecursiveOnlinePhaseEstimator(
            n_dims_estimand_pos             = n_dims_estimand_pos,
            listening_time                  = listening_time,
            discarded_time                  = discarded_time,
            min_duration_first_pseudoperiod= min_duration_first_quasiperiod,
            look_behind_pcent               = look_behind_pcent,
            look_ahead_pcent                = look_ahead_pcent,
            time_const_lowpass_filter_pos = time_const_lowpass_filter_estimand_pos,
            time_const_lowpass_filter_phase = time_const_lowpass_filter_phase,
            is_use_baseline                 = is_use_baseline,
            baseline_pos_loop               = baseline_pos_loop,
            time_step_baseline              = time_step_baseline,
            ref_frame_estimand_points       = ref_frame_estimand_points,
            ref_frame_baseline_points       = ref_frame_baseline_points,
            is_use_elapsed_time             = True,
        )
        phase_estimand_online = np.full(n_time_instants, None)
        for i_t in range(n_time_instants - 1):
            phase_estimand_online[i_t] = phase_estimator.update_estimator(estimand_pos_signal[i_t,:], time_signal[i_t])
        # Offline estimator
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
        interval_between_pca = 3
        phase_estimator_online_PCA = Phase_estimator_pca_online(window_pca,interval_between_pca)
        phase_estimand_online_PCA = np.full(n_time_instants, None)
        for i_t in range(n_time_instants - 1):
            print(i_t)
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
    
        loop = 17  # set the first loop you want to plot
        n_loop = 2  # set how many loops you want to plot

        index_start = np.argmax( time_signal >= phase_estimator.delimiter_time_instants[loop-1])
        plot_window = np.argmax( time_signal > phase_estimator.delimiter_time_instants[loop+n_loop-1])
        

        # Figure error
        # ------------------------------------------------
        real_phase = np.array(real_phase, dtype=float)
        colors = cmap.colors
        phase_estimand_online = np.nan_to_num(np.array(phase_estimand_online, dtype=float), nan=0.0)
        phase_estimand_online_PCA = np.nan_to_num(np.array(phase_estimand_online_PCA, dtype=float), nan=0.0)
        phase_estimand_offline = np.nan_to_num(np.array(phase_estimand_offline, dtype=float), nan=0.0)
        error_online = np.abs(np.angle(np.exp(1j * (real_phase[index_start:plot_window] - phase_estimand_online[index_start:plot_window]))))
        error_online_PCA = np.abs(np.angle(np.exp(1j * (real_phase[index_start:plot_window] - phase_estimand_online_PCA[index_start:plot_window]))))
        error_offline = np.abs(np.angle(np.exp(1j * (real_phase[index_start:plot_window] - phase_estimand_offline[index_start:plot_window]))))
        # Ensure the errors are stored
        all_errors_online = np.vstack(error_online)
        all_errors_online_PCA = np.vstack(error_online_PCA)
        all_errors_offline = np.vstack(error_offline)

        plt.figure(figsize=(10, 3))
        plt.plot(time_signal[index_start:plot_window], real_phase[index_start:plot_window], label='Benchmark', linewidth=1.5,linestyle='--',color=colors[4])
        plt.plot(time_signal[index_start:plot_window], phase_estimand_online_PCA[index_start:plot_window], label='PCA-T',color=colors[5])
        plt.plot(time_signal[index_start:plot_window], phase_estimand_offline[index_start:plot_window], label='PCA-H',color=colors[1])
        plt.plot(time_signal[index_start:plot_window], phase_estimand_online[index_start:plot_window], label='ROPE',color=colors[2])
        plt.xlabel("Time [s]")
        plt.ylabel("Phase [rad]")
        plt.title(f"Phases {shape} iteration: {iteration}")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 3))
        plt.plot(time_signal[index_start:plot_window], error_online_PCA, label='PCA-T',color=colors[5])
        plt.plot(time_signal[index_start:plot_window], error_offline, label='PCA-H',color=colors[1])
        plt.plot(time_signal[index_start:plot_window], error_online, label='ROPE',color=colors[2])
        plt.xlabel("Time [s]")
        plt.ylabel("Phase [rad]")
        plt.title(f"Error {shape} iteration: {iteration}")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 3))
        plt.plot(time_signal[index_start:plot_window], estimand_pos_signal[index_start:plot_window, 0], label='ECG', color=colors[0])
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title(f"Signal {shape} iteration: {iteration}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Compute cumulative mean and variance
    cum_mean_online = np.cumsum(np.mean(all_errors_online, axis=0)) / np.arange(1, len(all_errors_online[0]) + 1)
    cum_var_online = np.cumsum(np.var(all_errors_online, axis=0)) / np.arange(1, len(all_errors_online[0]) + 1)
    
    cum_mean_online_PCA = np.cumsum(np.mean(all_errors_online_PCA, axis=0)) / np.arange(1, len(all_errors_online_PCA[0]) + 1)
    cum_var_online_PCA = np.cumsum(np.var(all_errors_online_PCA, axis=0)) / np.arange(1, len(all_errors_online_PCA[0]) + 1)
    
    cum_mean_offline = np.cumsum(np.mean(all_errors_offline, axis=0)) / np.arange(1, len(all_errors_offline[0]) + 1)
    cum_var_offline = np.cumsum(np.var(all_errors_offline, axis=0)) / np.arange(1, len(all_errors_offline[0]) + 1)
    
    # Store results in dictionary
    error_dict[shape]['ROPE'] = (cum_mean_online, cum_var_online)
    error_dict[shape]['PCA_T'] = (cum_mean_online_PCA, cum_var_online_PCA)
    error_dict[shape]['PCA_H'] = (cum_mean_offline, cum_var_offline)

# Plot bar plots with error bars
fig, ax = plt.subplots(figsize=(6,3))
plt.rcParams.update({'font.size': 10, 'font.family': 'Times New Roman'})
ax.grid(True, axis='y')
shapes_labels = list(shapes_name)
x = np.arange(len(shapes_labels)) * 3  # Grouping by shape, spacing out clusters
width = 0.3  # Width of bars

means_online = [np.mean(error_dict[shape]['ROPE'][0]) for shape in shapes]
stds_online = [np.mean(error_dict[shape]['ROPE'][1]) for shape in shapes]

means_online_PCA = [np.mean(error_dict[shape]['PCA_T'][0]) for shape in shapes]
stds_online_PCA = [np.mean(error_dict[shape]['PCA_T'][1]) for shape in shapes]

means_offline = [np.mean(error_dict[shape]['PCA_H'][0]) for shape in shapes]
stds_offline = [np.mean(error_dict[shape]['PCA_H'][1]) for shape in shapes]

ax.bar(x - width, means_online, width, yerr=stds_online, label='ROPE', capsize=5, color=colors[2])
ax.bar(x, means_online_PCA, width, yerr=stds_online_PCA, label='PCA-T', capsize=5, color=colors[5])
ax.bar(x + width, means_offline, width, yerr=stds_offline, label='PCA-H', capsize=5, color=colors[1])

ax.set_xticks(x)
ax.set_xticklabels(shapes_labels)
ax.set_xlim(x[0] - 2, x[-1] + 2)  # margine laterale
ax.legend()

plt.tight_layout()
plt.show()

print("\n=== Error Statistics ===")
for shape in shapes:
    print(f"\nShape: {shape.capitalize()}")
    for method, label in zip(['ROPE', 'PCA_T', 'PCA_H'], ['ROPE', 'PCA-T', 'PCA-H']):
        mean = np.mean(error_dict[shape][method][0])
        var = np.mean(error_dict[shape][method][1])
        print(f"  {label}: Mean Error = {mean:.4f}, Variance = {var:.6f}")





