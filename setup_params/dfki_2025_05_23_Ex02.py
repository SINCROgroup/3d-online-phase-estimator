file_path_estimand = r"data\dfki_mocap_excerises_2025-05-23\CSV\Ex02_05.csv"
# iterations 5, 10, 12 very well done
# iterations 7, 11 poorly done

col_names_pos_estimand  = ["X25", "Y25", "Z25", "X21", "Y21", "Z21", "X37", "Y37", "Z37", "X31", "Y31", "Z31"] # LWrist, RWrist, LHeel, RHeel

discarded_time = 2
look_behind_pcent = 5
look_ahead_pcent = 40
listening_time = 20
time_const_lowpass_filter_estimand_pos = 0.1
time_const_lowpass_filter_phase = 0.2
rows_to_skip_estimand = list(range(0, 4))
time_step = 1/30

is_update_comparison_loop = True
is_use_baseline = False;  time_step_baseline = None
is_use_elapsed_time = False

