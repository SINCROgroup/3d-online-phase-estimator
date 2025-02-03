import numpy as np
from numpy.matlib import zeros

from fun_first_period_estimation import compute_signal_period_autocorrelation
from scipy.signal import find_peaks

class OnlineMultidimPhaseEstimator:
    def __init__(self,
                 n_dims_estimand_pos: int,
                 step_time                      = 0.01,
                 discarded_time                 = 5,
                 listening_time                 = 15,
                 min_duration_first_quasiperiod = 1,
                 look_behind_pcent              = 0,
                 look_ahead_pcent               = 25,
                 is_use_baseline                = False,
                 baseline_pos_loop              = None,
                 ref_frame_point_1              = None,
                 ref_frame_point_2              = None,
                 ref_frame_point_3              = None):

        # Initialization from arguments
        self.n_dim                       = n_dims_estimand_pos
        self.step_time                   = step_time             # [s]
        self.discarded_time              = discarded_time              # initial waiting time interval [s]
        self.listening_time              = listening_time # time interval in witch to estimate the first period [s]
        self.look_ahead_pcent            = look_ahead_pcent  # % of the last completed period before the last nearest point on which estimate the new phase
        self.look_behind_pcent           = look_behind_pcent # % of the last completed period after the last nearest point on which estimate the new phase
        self.is_use_baseline             = is_use_baseline
        self.min_duration_quasiperiod    = min_duration_first_quasiperiod # [s]

        self.initial_time                = 0              # start time instant of fase computation [s]
        self.max_diff_len_new_loop_pcent = 30             # difference in length of the new reference (vector length) compared to the old one              , expressed as a percentage of the old length, is accepted.
        self.is_first_loop_estimated = False
        self.epsilon                 = np.pi
        self.offset                  = 0
        self.curr_phase              = None
        self.prev_phase              = None
        self.idx_curr_time_loop       = 0
        self.idx_curr_phase_in_latest_loop = 0
        self.prev_pos = np.zeros(self.n_dim)

        self.max_length_loop         = 1000
        self.latest_pos_loop         = np.zeros((self.max_length_loop, 2*self.n_dim))   # TODO test making this not bounded bt max length
        self.new_loop                = np.zeros((self.max_length_loop, 2*self.n_dim))   # TODO make this something which is appended each time

        self.pos_signal              = []
        self.vel_signal              = []
        self.local_time_vec          = []
        self.delimiter_time_instants = []

        self.len_last_period_discarded = 0
        self.look_ahead_range          = 0
        self.look_behind_range         = 0

        if is_use_baseline:
            assert n_dims_estimand_pos == 3, "Tethered mode can be used only with n_dim = 3"
            assert baseline_pos_loop is not None, "Tethered mode was required but baseline_pos_loop was not provided"
            assert ref_frame_point_1 is not None, "Tethered mode was required but ref_frame_point_1 was not provided"
            assert ref_frame_point_2 is not None, "Tethered mode was required but ref_frame_point_2 was not provided"
            assert ref_frame_point_3 is not None, "Tethered mode was required but ref_frame_point_3 was not provided"

            self.baseline_pos_loop = baseline_pos_loop.copy()
            self.ref_frame_point_1 = ref_frame_point_1.copy()
            self.ref_frame_point_2 = ref_frame_point_2.copy()
            self.ref_frame_point_3 = ref_frame_point_3.copy()


    def compute_phase(self, curr_pos, curr_time) -> float:
        if not self.local_time_vec:  self.initial_time = curr_time  # initialize initial_time

        self.local_time_vec.append(curr_time - self.initial_time)

        if  self.local_time_vec[-1] > self.discarded_time:
            self.pos_signal.append(curr_pos)
            self.step_time = self.local_time_vec[-1] - self.local_time_vec[-2]
            self.vel_signal.append((curr_pos - self.prev_pos) / self.step_time)

        self.prev_pos = curr_pos
                
        if not self.is_first_loop_estimated:
            if self.local_time_vec[-1] > self.discarded_time + self.listening_time:
                self.latest_pos_loop = compute_signal_period_autocorrelation(
                    pos_signal               = self.pos_signal,
                    vel_signal               = self.vel_signal,
                    local_time_vec           = self.local_time_vec,
                    min_duration_quasiperiod = self.min_duration_quasiperiod)

                self.is_first_loop_estimated = True
                self.look_ahead_range  = max(1, int(len(self.latest_pos_loop) * self.look_ahead_pcent  / 100))   # range_post is the number of points after the last nearest point on which estimate the new phase
                self.look_behind_range = max(1, int(len(self.latest_pos_loop) * self.look_behind_pcent / 100))

                if self.is_use_baseline:  self.compute_phase_offset()

                # estimate phase for the first loop
                for i in range(len(self.latest_pos_loop), len(self.pos_signal) - 1):
                    curr_kinematics = np.concatenate((self.pos_signal[i], self.vel_signal[i]))
                    self.compute_phase_internal(curr_kinematics)
                    self.update_latest_loop(curr_kinematics)

        if len(self.pos_signal) > 0:
            curr_kinematics = np.concatenate((self.pos_signal[-1], self.vel_signal[-1]))
        else:
            curr_kinematics = zeros(2*self.n_dim)

        self.compute_phase_internal(curr_kinematics)

        if self.is_first_loop_estimated:  self.update_latest_loop(curr_kinematics)

        return np.mod(self.curr_phase + self.offset, 2*np.pi)


    def update_latest_loop(self, curr_kinematics): # receives position and velocity as input and updates the vector of the last complete period when the phase completes a full cycle.
        """checks whether it is necessary to update the latest loop, and, if so, does it"""
        def update_latest_loop_and_ranges() -> None:
            self.latest_pos_loop   = self.new_loop[0:self.idx_curr_time_loop, :]
            self.look_ahead_range  = max(1, int(len(self.latest_pos_loop) * self.look_ahead_pcent  / 100))
            self.look_behind_range = max(1, int(len(self.latest_pos_loop) * self.look_behind_pcent / 100))

        if self.curr_phase - self.prev_phase < -self.epsilon:   # a quasiperiodicity window ended
            self.delimiter_time_instants.append(float(self.local_time_vec[-1] + self.initial_time))
            length_new_loop = self.idx_curr_time_loop + 1
            # if the difference in length between new loop and previous loop is smaller than the range set by user
            if abs(length_new_loop - len(self.latest_pos_loop)) < len(self.latest_pos_loop)*self.max_diff_len_new_loop_pcent/100:
                update_latest_loop_and_ranges()

            # if the difference in length between new loop and previous discarded loop is smaller than the range set by user   # TODO MC: I don't get the rationale behind this logic
            elif abs(length_new_loop - self.len_last_period_discarded) < self.len_last_period_discarded*self.max_diff_len_new_loop_pcent/100:
                update_latest_loop_and_ranges()
                self.len_last_period_discarded = 0

            else:
                self.len_last_period_discarded = self.idx_curr_time_loop + 1

            # reinitialize new_loop
            self.new_loop = np.zeros((self.max_length_loop, 2*self.n_dim))
            self.idx_curr_time_loop = 0

        # append current kinematics to new loop
        if self.idx_curr_time_loop < self.max_length_loop:
            self.new_loop[self.idx_curr_time_loop, :] = curr_kinematics
            self.idx_curr_time_loop += 1


    def compute_phase_internal(self, curr_kinematics):
        if self.is_first_loop_estimated:
            len_latest_loop = len(self.latest_pos_loop)
            if self.idx_curr_phase_in_latest_loop - self.look_behind_range < 1:
                #    loop: [part_1 - - - - - - - - part_2]
                loop_part_1 = self.latest_pos_loop[0 : self.idx_curr_phase_in_latest_loop + self.look_ahead_range]
                loop_part_2 = self.latest_pos_loop[len_latest_loop - self.look_behind_range + self.idx_curr_phase_in_latest_loop : len_latest_loop]
                loop_for_search = np.vstack((loop_part_1, loop_part_2))

                idxs_part_1 = np.arange(1, self.idx_curr_phase_in_latest_loop + self.look_ahead_range + 1)
                idxs_part_2 = np.arange(len_latest_loop - self.look_behind_range + self.idx_curr_phase_in_latest_loop, len_latest_loop + 1)
                idxs_loop_for_search = np.concatenate((idxs_part_1, idxs_part_2))
            elif self.idx_curr_phase_in_latest_loop + self.look_ahead_range > len_latest_loop:
                #    loop: [part_2 - - - - - - - - part_1]
                loop_part_1 = self.latest_pos_loop[self.idx_curr_phase_in_latest_loop - self.look_behind_range:len_latest_loop]
                loop_part_2 = self.latest_pos_loop[0:self.idx_curr_phase_in_latest_loop + self.look_ahead_range - len_latest_loop]
                loop_for_search = np.vstack((loop_part_1, loop_part_2))

                idxs_part_1 = np.arange(self.idx_curr_phase_in_latest_loop - self.look_behind_range, len_latest_loop + 1)
                idxs_part_2 = np.arange(1, self.idx_curr_phase_in_latest_loop + self.look_ahead_range - len_latest_loop + 1)
                idxs_loop_for_search = np.concatenate((idxs_part_1, idxs_part_2))
            else:
                #   loop: [- - - - - - single part - - - - -]
                loop_for_search = self.latest_pos_loop[self.idx_curr_phase_in_latest_loop - self.look_behind_range:self.idx_curr_phase_in_latest_loop + self.look_ahead_range]
                idxs_loop_for_search = np.arange(self.idx_curr_phase_in_latest_loop - self.look_behind_range, self.idx_curr_phase_in_latest_loop + self.look_ahead_range + 1)

            squared_err_pos = loop_for_search.copy()
            squared_err_vel = loop_for_search.copy()
            for i in range(len(loop_for_search)):
                tmp = (loop_for_search[i] - curr_kinematics) ** 2
                squared_err_pos[i] = np.concatenate((np.ones(self.n_dim), np.zeros(self.n_dim))) * tmp
                squared_err_vel[i] = np.concatenate((np.zeros(self.n_dim), np.ones(self.n_dim))) * tmp
            distances_pos = np.sqrt(np.sum((squared_err_pos), axis=1))  # position error norm
            distances_vel = np.sqrt(np.sum((squared_err_vel), axis=1))  # velocity error norm
            distances_pos = distances_pos/max(distances_pos)      # normalized position error norm
            distances_vel = distances_vel/max(distances_vel)      # normalized velocity error norm
            distances = distances_pos + distances_vel           # normalized error
            index_min_distance = np.argmin(distances)       # minimum distance point
            self.idx_curr_phase_in_latest_loop = idxs_loop_for_search[index_min_distance]
            self.prev_phase = self.curr_phase
            self.curr_phase = (2 * np.pi * self.idx_curr_phase_in_latest_loop) / len_latest_loop
            if self.curr_phase - self.prev_phase > self.epsilon:  # Avoid 0 to 2pi jumps
                 self.curr_phase = self.prev_phase
            if self.curr_phase - self.prev_phase < 0 and self.prev_phase - self.curr_phase < self.epsilon:
                self.curr_phase = self.prev_phase
        
        else:
            self.prev_phase = 0
            self.curr_phase = 0

    
    def compute_phase_offset(self) -> None:
        x_axis = (self.ref_frame_point_2 - self.ref_frame_point_1) / np.linalg.norm(self.ref_frame_point_2 - self.ref_frame_point_1)
        z_vector = self.ref_frame_point_3 - self.ref_frame_point_2
        z_axis = z_vector - np.dot(z_vector, x_axis) * x_axis 
        z_axis = z_axis / np.linalg.norm(z_axis)  
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis])
        
        rotated_loop = self.latest_pos_loop[:, 0:3] @ rotation_matrix.T
        #rotated_loop = self.latest_pos_loop[:, 0:self.n_dim] @ np.kron(np.eye(int(self.n_dim/3)), rotation_matrix.T)

        centroid = np.mean(rotated_loop, axis=0)
        rotated_centered_loop = rotated_loop - centroid

        scale_factors = np.std(self.baseline_pos_loop, axis=0) / np.std(rotated_centered_loop, axis=0)
        scale_factors[np.isnan(scale_factors)] = 1 
        scaled_rotated_centered_loop = rotated_centered_loop * scale_factors

        time_step_baseline = 0.01
        p = scaled_rotated_centered_loop[0, :]
        vel=(scaled_rotated_centered_loop[1, :]-scaled_rotated_centered_loop[0, :])/self.step_time
        baseline_vel_loop = np.gradient(self.baseline_pos_loop, np.arange(0, len(self.baseline_pos_loop) * time_step_baseline, time_step_baseline), axis=0)
        position_norm = self.baseline_pos_loop.copy()
        vel_norm = baseline_vel_loop.copy()
        for i in range(len(self.baseline_pos_loop)):
            position_norm[i] = (self.baseline_pos_loop[i] - p) ** 2
            vel_norm[i] = (baseline_vel_loop[i] - vel) ** 2
        distances_p = np.sqrt(np.sum((position_norm), axis=1))
        distances_v = np.sqrt(np.sum((vel_norm), axis=1))
        distances_p = distances_p/max(distances_p)
        distances_v = distances_v/max(distances_v)
        distances = distances_p + distances_v
        index = np.argmin(distances)
        self.offset = (2 * np.pi * index) / len(self.baseline_pos_loop)