file_path_estimand = r"data\ideal_oscillator\sine_wave_data.csv"; 
col_names_pos_estimand = ["sine","cosine"];
look_behind_pcent = 5; look_ahead_pcent = 15; 
listening_time = 5; 
discarded_time = 5; 
time_const_lowpass_filter_estimand_pos = 0

rows_to_skip_estimand = list(range(0, 0))
time_step             = 1/24

is_use_baseline = False;
time_step_baseline = None