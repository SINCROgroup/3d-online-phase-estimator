file_path_estimand = r"data\dfki_mocap_excerises_2025-05-23\CSV\Ex02_baseline_without_phase_repeated.csv"

col_names_pos_estimand  = ["RWrist.X","RWrist.Y","RWrist.Z",
						   "LWrist.X","LWrist.Y","LWrist.Z",
						   "RHeel.X","RHeel.Y","RHeel.Z",
						   "LHeel.X","LHeel.Y","LHeel.Z",
						   "LHeel.X","LHeel.Y","LHeel.Z",
						   "Hip.X","Hip.Y","Hip.Z"]

discarded_time = 0
look_behind_pcent = 5
look_ahead_pcent = 40
listening_time = 30
time_const_lowpass_filter_estimand_pos = 0.1
time_const_lowpass_filter_phase = 0.2
rows_to_skip_estimand = list(range(0, 0))
time_step = 1/30
is_use_baseline = False;  time_step_baseline = None
is_use_elapsed_time = False

