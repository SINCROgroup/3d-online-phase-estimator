file_path_estimand = r"data\dfki_mocap_excerises_2025-03-25\exercise_1\positions.csv"
col_names_pos_estimand  = ["X22", "Y22", "Z22", "X26", "Y26", "Z26"] # Rshoulder and Lshoulder
discarded_time = 5
look_behind_pcent = 10
look_ahead_pcent = 40
listening_time = 20
time_const_lowpass_filter_estimand_pos = 0.1
rows_to_skip_estimand = list(range(0, 4))
time_step = 1/30
is_use_baseline = False;  time_step_baseline = None