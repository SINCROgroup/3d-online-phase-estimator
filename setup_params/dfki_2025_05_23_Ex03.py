file_path_estimand = r"data\dfki_mocap_excerises_2025-05-23\CSV\Ex03_09.csv"

col_names_pos_estimand  = ["X1", "Y1", "Z1",     # Hip
						   "X32", "Y32", "Z32",  # LHip
						   "X26", "Y26", "Z26"]  # RHip

discarded_time = 2
look_behind_pcent = 5
look_ahead_pcent = 40
listening_time = 12
time_const_lowpass_filter_estimand_pos = 0.1
time_const_lowpass_filter_phase = 0.2
rows_to_skip_estimand = list(range(0, 4))
time_step = 1/30
is_use_baseline = False;  time_step_baseline = None
is_use_elapsed_time = False