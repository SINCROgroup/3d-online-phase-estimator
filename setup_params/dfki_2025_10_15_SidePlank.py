# iterations x, x, x very well done
# iterations x, x poorly done

file_path_estimand = r"data\dfki_mocap_excerises_2025-10-15\CSV\20251017_SidePlank_10.csv"
rows_to_skip_estimand = list(range(0, 4))
col_names_pos_estimand  = ["X1", "Y1", "Z1", "X18", "Y18", "Z18", "X22", "Y22", "Z22"]  # pelvis, thigh left, thigh right


discarded_time = 15
look_behind_pcent = 5
look_ahead_pcent = 50
listening_time = 18
time_const_lowpass_filter_estimand_pos = 0.1
time_const_lowpass_filter_phase = 0.2
time_step = 1/30

is_update_comparison_loop = True
is_use_elapsed_time = True

is_use_baseline = True
file_path_baseline = r"data\dfki_mocap_excerises_2025-10-15\CSV\20251017_SidePlank_baseline.csv"
col_names_pos_baseline  = ["pelvis.X", "pelvis.Y", "pelvis.Z", "thigh_l.X", "thigh_l.Y", "thigh_l.Z", "thigh_r.X", "thigh_r.Y", "thigh_r.Z"]
time_step_baseline = 1/30