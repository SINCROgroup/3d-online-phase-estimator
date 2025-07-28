file_path_estimand = r"data\cyens_mocap_2025-05-08\Ex2-Sub3-0_worldpos.csv";
col_names_pos_estimand = ["RightLeg.X","RightLeg.Y","RightLeg.Z", "LeftLeg.X","LeftLeg.Y","LeftLeg.Z",
                          "RightFoot.X","RightFoot.Y","RightFoot.Z", "LeftFoot.X","LeftFoot.Y","LeftFoot.Z",
                          "LeftArm.X", "LeftArm.Y", "LeftArm.Z", "LeftArm.X", "LeftArm.Y", "LeftArm.Z",
                          "RightHand.X","RightHand.Y","RightHand.Z", "LeftHand.X","LeftHand.Y","LeftHand.Z"]
listening_time = 30
time_const_lowpass_filter_estimand_pos = 0.1
time_const_lowpass_filter_phase        = 0.3
look_behind_pcent = 5; look_ahead_pcent = 25;

rows_to_skip_estimand = list(range(0, 0))

is_update_comparison_loop = False
is_use_baseline = False
is_use_elapsed_time = True

time_step = 0.016667