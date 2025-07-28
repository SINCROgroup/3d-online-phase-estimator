file_path_estimand = r"data\cyens_mocap_2025-05-08\Ex3-Sub1-0_worldpos.csv";
col_names_pos_estimand = ["Hips.X","Hips.Y","Hips.Z",
                          "LeftUpLeg.X","LeftUpLeg.Y","LeftUpLeg.Z",
                          "RightUpLeg.X","RightUpLeg.Y","RightUpLeg.Z"]
time_step = 0.016667

discarded_time = 5
listening_time = 20
time_const_lowpass_filter_estimand_pos = 0.1
time_const_lowpass_filter_phase        = 0.3
look_behind_pcent = 5; look_ahead_pcent = 25

rows_to_skip_estimand = list(range(0, 0))

is_update_comparison_loop = True
is_use_baseline = False
is_use_elapsed_time = True
