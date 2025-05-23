file_path_estimand  = r"data\san_giovanni_2024-10-10\spiral_mc_1.csv"; listening_time = 10
# file_path_estimand  = r"data\san_giovanni_2024-10-10\clockwise_and_anticlockwise_circle_mc_1.csv"; listening_time = 15
# file_path_estimand  = r"data\san_giovanni_2024-10-10\macarena_mc_2.csv"; listening_time = 15

rows_to_skip_estimand  = [0, 1, 2] + list(range(4, 40))
col_names_pos_estimand = ['TX.3', 'TY.3', 'TZ.3']
time_step              = 0.01  # [s]

is_use_baseline = True
time_step_baseline  = 0.01
# file_path_baseline  = r"data\san_giovanni_2024-10-10\spiral_ref.csv"       # TODO edit point. path A
file_path_baseline  = r"data\san_giovanni_2024-10-10\spiral_baseline.csv"    # TODO edit point. path B
col_names_pos_baseline = ['x', 'y', 'z']
col_names_ref_frame_estimand_points = [['TX', 'TY', 'TZ'], ['TX.2', 'TY.2', 'TZ.2'], ['TX.1', 'TY.1', 'TZ.1']]   # belly, # right chest, # left chest
col_names_ref_frame_baseline_points = [['p1_x','p1_y','p1_z'], ['p2_x','p2_y','p2_z'], ['p3_x','p3_y','p3_z']]
