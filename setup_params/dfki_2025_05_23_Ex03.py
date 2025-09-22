# file_path_estimand = r"data\dfki_mocap_excerises_2025-05-23\CSV\Ex03_09.csv"
# rows_to_skip_estimand = list(range(0, 4))
# col_names_pos_estimand  = ["X1", "Y1", "Z1",     # Hip
# 						   "X32", "Y32", "Z32",  # LHip
# 						   "X26", "Y26", "Z26"]  # RHip

file_path_estimand = r"data\dfki_mocap_excerises_2025-05-23\CSV\Ex03_baseline_without_phase_repeated.csv"
rows_to_skip_estimand = list(range(0, 0))
col_names_pos_estimand  = ["Hip.X", "Hip.Y", "Hip.Z", "LHip.X", "LHip.Y", "LHip.Z", "RHip.X", "RHip.Y", "RHip.Z"]

discarded_time = 1
look_behind_pcent = 5
look_ahead_pcent = 40
listening_time = 12
time_const_lowpass_filter_estimand_pos = 0.1
time_const_lowpass_filter_phase = 0.2
time_step = 1/30
is_use_elapsed_time = False
is_update_comparison_loop = True

is_use_baseline = True
file_path_baseline = r"data\dfki_mocap_excerises_2025-05-23\CSV\Ex03_baseline.csv"
col_names_pos_baseline  = ["Hip.X", "Hip.Y", "Hip.Z", "LHip.X", "LHip.Y", "LHip.Z", "RHip.X", "RHip.Y", "RHip.Z"]
time_step_baseline = 1/30