file_path_estimand = r"data\tests_unreal\2025_06_27\ex02_positions_history.csv"; 
col_names_pos_estimand = ["col1","col4","col7",]
look_behind_pcent = 5; look_ahead_pcent = 40; 
listening_time = 5; 
discarded_time = 1; 
time_const_lowpass_filter_estimand_pos = 0
time_const_lowpass_filter_estimand_phase = 0

rows_to_skip_estimand = list(range(0, 0))
time_step             = 0.1

is_update_comparison_loop = False
is_use_baseline = False
time_step_baseline = None