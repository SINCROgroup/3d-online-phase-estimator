import numpy as np
from fun_first_period_estimation import compute_signal_period_autocorrelation
from scipy.signal import find_peaks

class Online3DPhaseEstimator:
    def __init__(self,
                 step_time           = 0.01,
                 look_behind_pcent   = 0,
                 look_ahead_pcent    = 25,
                 wait_time           = 5,
                 listening_time      = 15,
                 min_duration_period = None,
                 baseline_pos_loop   = None,
                 ref_frame_point_1   = None,
                 ref_frame_point_2   = None,
                 ref_frame_point_3   = None):

        # Initialization from arguments
        self.step_time           = step_time      # [s]
        self.initial_time        = 0              # start time instant of fase computation [s]
        self.wait_time           = wait_time      # initial waiting time interval [s]
        self.listening_time      = listening_time # time interval in witch to estimate the first period [s]
        self.min_length_quasiperiod = 1              # minimum time duration of periods [s]
        self.look_ahead_pcent    = look_ahead_pcent  # % of the last completed period before the last nearest point on which estimate the new phase
        self.look_behind_pcent   = look_behind_pcent # % of the last completed period after the last nearest point on which estimate the new phase
        self.diff_len_new_ref    = 30             # difference in length of the new reference (vector length) compared to the old one                          , expressed as a percentage of the old length, is accepted.

        self.max_length_vec_period     = 1000
        self.is_first_loop_estimated = False
        self.epsilon                   = np.pi
        self.offset                    = 0
        self.phase                     = np.zeros(2)
        self.curr_phase = None
        self.prev_phase = None
        self.latest_pos_loop        = np.zeros((self.max_length_vec_period, 6))
        self.new_period                = np.zeros((self.max_length_vec_period, 6))

        # Initialization of empty attributes
        self.pos_x_signal     = []
        self.pos_y_signal     = []
        self.pos_z_signal     = []
        self.vel_x_signal = []
        self.vel_y_signal = []
        self.vel_z_signal = []
        self.local_time_vec  = []

        self.baseline_pos_loop = None
        self.point_1   = None
        self.point_2   = None
        self.point_3   = None

        self.counter       = 0
        self.min_index_pre = 0
        self.prev_pos = [0, 0, 0]

        self.len_last_period_discarded = 0
        self.look_ahead_range          = 0
        self.look_behind_range         = 0

        if min_duration_period is not None:
            self.min_length_quasiperiod = min_duration_period
        if baseline_pos_loop is not None:
            self.baseline_pos_loop=baseline_pos_loop.copy()
            if ref_frame_point_1 is not None and ref_frame_point_2 is not None and ref_frame_point_3 is not None:
                self.point_1 = ref_frame_point_1
                self.point_2 = ref_frame_point_2
                self.point_3 = ref_frame_point_3
            else:
                raise ValueError("left_chest, right_chest and belly must be different from None ")


    def set_position(self, curr_pos, curr_time) -> float:  # TODO change name of the function?
        if not self.local_time_vec:  self.initial_time = curr_time  # initialize initial_time

        self.local_time_vec.append(curr_time - self.initial_time)

        if  self.local_time_vec[-1] > self.wait_time:
            self.pos_x_signal.append(curr_pos[0])         # TODO should be a numpy matrix
            self.pos_y_signal.append(curr_pos[1])
            self.pos_z_signal.append(curr_pos[2])
            self.step_time = self.local_time_vec[-1] - self.local_time_vec[-2]
            self.vel_x_signal.append((curr_pos[0] - self.prev_pos[0]) / self.step_time)
            self.vel_y_signal.append((curr_pos[1] - self.prev_pos[1]) / self.step_time)
            self.vel_z_signal.append((curr_pos[2] - self.prev_pos[2]) / self.step_time)

        self.prev_pos = curr_pos
                
        if not self.is_first_loop_estimated:
            if self.local_time_vec[-1] > self.wait_time + self.listening_time:
                self.latest_pos_loop = compute_signal_period_autocorrelation(
                    pos_x_signal           = self.pos_x_signal,
                    pos_y_signal           = self.pos_y_signal,
                    pos_z_signal           = self.pos_z_signal,
                    vel_x_signal           = self.vel_x_signal,
                    vel_y_signal           = self.vel_y_signal,
                    vel_z_signal           = self.vel_z_signal,
                    local_time_vec         = self.local_time_vec,
                    min_length_quasiperiod = self.min_length_quasiperiod)

                self.is_first_loop_estimated = True
                self.look_ahead_range  = max(1, int(len(self.latest_pos_loop) * self.look_ahead_pcent  / 100))   # range_post is the number of points after the last nearest point on which estimate the new phase
                self.look_behind_range = max(1, int(len(self.latest_pos_loop) * self.look_behind_pcent / 100))

                if self.baseline_pos_loop is not None:  self.compute_phase_offset()   # TODO introduce a string mode_algorithm that can be tethered or un tethered. Should be put elsewhere

                # estimate phase for the first loop
                for i in range(len(self.latest_pos_loop), len(self.pos_x_signal) - 1):
                    curr_kinematics = np.array([self.pos_x_signal[i], self.pos_y_signal[i], self.pos_z_signal[i], self.vel_x_signal[i], self.vel_y_signal[i], self.vel_z_signal[i]])
                    self.compute_phase(curr_kinematics)
                    self.estimate_period(curr_kinematics)

        if len(self.pos_x_signal) > 0:
            curr_kinematics = np.array([self.pos_x_signal[-1], self.pos_y_signal[-1], self.pos_z_signal[-1], self.vel_x_signal[-1], self.vel_y_signal[-1], self.vel_z_signal[-1]])
        else:
            curr_kinematics = np.array([0,0,0,0,0,0])

        self.compute_phase(curr_kinematics) # compute the phase

        if self.is_first_loop_estimated:  self.estimate_period(curr_kinematics)

        return np.mod(self.curr_phase + self.offset, 2*np.pi)


    def estimate_period(self, curr_kinematics): # receives position and velocity as input and updates the vector of the last complete period when the phase completes a full cycle.
        if self.curr_phase - self.prev_phase < -self.epsilon:
            if abs(self.counter-len(self.latest_pos_loop)) < len(self.latest_pos_loop)*self.diff_len_new_ref/100: # Ensure that the difference in length between the new period and the previous one is not greater than the range set by the user.
                self.latest_pos_loop = self.new_period[0:self.counter, :]
                self.look_ahead_range = int(len(self.latest_pos_loop) * self.look_ahead_pcent / 100)
                if self.look_ahead_range==0:  self.look_ahead_range=1
                self.look_behind_range = int(len(self.latest_pos_loop) * self.look_behind_pcent / 100)
                if self.look_behind_range == 0:  self.look_behind_range = 1

            elif abs(self.counter-self.len_last_period_discarded) < self.len_last_period_discarded*self.diff_len_new_ref/100:
                self.latest_pos_loop = self.new_period[0:self.counter, :]
                self.look_ahead_range = int(len(self.latest_pos_loop) * self.look_ahead_pcent / 100)
                if self.look_ahead_range==0:
                    self.look_ahead_range=1
                self.look_behind_range = int(len(self.latest_pos_loop) * self.look_behind_pcent / 100)
                if self.look_behind_range == 0:
                    self.look_behind_range = 1
                self.len_last_period_discarded = 0

            else:
                self.len_last_period_discarded = self.counter

            self.new_period = np.zeros((self.max_length_vec_period, 6))
            self.counter = 0
        if self.counter < self.max_length_vec_period:
            self.new_period[self.counter,:] = curr_kinematics
            self.counter += 1


    def compute_phase(self, p):
        if self.is_first_loop_estimated:
            length_reference = len(self.latest_pos_loop)
            if self.min_index_pre - self.look_behind_range < 1:
                ref_part1 = self.latest_pos_loop[0:self.min_index_pre + self.look_ahead_range]
                ref_part2 = self.latest_pos_loop[length_reference - self.look_behind_range + self.min_index_pre:length_reference]
                ref = np.vstack((ref_part1, ref_part2))

                index_part1 = np.arange(1, self.min_index_pre + self.look_ahead_range + 1)
                index_part2 = np.arange(length_reference - self.look_behind_range + self.min_index_pre, length_reference + 1)
                index = np.concatenate((index_part1, index_part2))
            elif self.min_index_pre + self.look_ahead_range > length_reference:
                ref_part1 = self.latest_pos_loop[self.min_index_pre - self.look_behind_range:length_reference]
                ref_part2 = self.latest_pos_loop[0:self.min_index_pre + self.look_ahead_range - length_reference]
                ref = np.vstack((ref_part1, ref_part2))

                index_part1 = np.arange(self.min_index_pre - self.look_behind_range, length_reference + 1)
                index_part2 = np.arange(1, self.min_index_pre + self.look_ahead_range - length_reference + 1)
                index = np.concatenate((index_part1, index_part2))
            else:
                ref = self.latest_pos_loop[self.min_index_pre - self.look_behind_range:self.min_index_pre + self.look_ahead_range]
                index = np.arange(self.min_index_pre - self.look_behind_range, self.min_index_pre + self.look_ahead_range + 1)

            ref_p = ref.copy()
            ref_v = ref.copy()
            for i in range(len(ref)):
                ref_p[i]=[1,1,1,0,0,0]*((ref[i]-p)**2)
                ref_v[i]=[0,0,0,1,1,1]*((ref[i]-p)**2)  
            distances_p = np.sqrt(np.sum((ref_p), axis=1))  # position error norm
            distances_v = np.sqrt(np.sum((ref_v), axis=1))  # velocity error norm
            distances_p = distances_p/max(distances_p)      # normalized position error norm
            distances_v = distances_v/max(distances_v)      # normalized velocity error norm
            distances = distances_p + distances_v           # normalized error
            index_min_distance = np.argmin(distances)       # minimum distance point
            self.min_index_pre = index[index_min_distance]
            self.prev_phase = self.curr_phase
            self.curr_phase = (2 * np.pi  * self.min_index_pre) / length_reference
            if self.curr_phase - self.prev_phase > self.epsilon:  # Avoid 0 to 2pi jumps
                 self.curr_phase = self.prev_phase
            if self.curr_phase - self.prev_phase < 0 and self.prev_phase - self.curr_phase < self.epsilon:
                self.curr_phase = self.prev_phase
        
        else:
            self.prev_phase = 0
            self.curr_phase = 0

    
    def compute_phase_offset(self) -> None:
        x_axis = (self.point_2 - self.point_1) / np.linalg.norm(self.point_2 - self.point_1)
        z_vector = self.point_3 - self.point_2
        z_axis = z_vector - np.dot(z_vector, x_axis) * x_axis 
        z_axis = z_axis / np.linalg.norm(z_axis)  
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis])
        
        rotated_ref = np.dot(self.latest_pos_loop[:, 0:3], rotation_matrix.T)
        
        centroid = np.mean(rotated_ref, axis=0)
        rotated_centered_ref = rotated_ref - centroid

        scale_factors = np.std(self.baseline_pos_loop, axis=0) / np.std(rotated_centered_ref, axis=0)
        scale_factors[np.isnan(scale_factors)] = 1 
        scaled_rotated_centered_ref = rotated_centered_ref * scale_factors

        p = scaled_rotated_centered_ref[0, :]
        vel=(scaled_rotated_centered_ref[1, :]-scaled_rotated_centered_ref[0, :])/self.step_time
        vel_ref=np.gradient(self.baseline_pos_loop, np.arange(0, len(self.baseline_pos_loop) * 0.01, 0.01), axis=0)
        position_norm = self.baseline_pos_loop.copy()
        vel_norm = vel_ref.copy()
        for i in range(len(self.baseline_pos_loop)):
            position_norm[i] = (self.baseline_pos_loop[i] - p) ** 2
            vel_norm[i] = (vel_ref[i] - vel) ** 2
        distances_p = np.sqrt(np.sum((position_norm), axis=1))
        distances_v = np.sqrt(np.sum((vel_norm), axis=1))
        distances_p = distances_p/max(distances_p)
        distances_v = distances_v/max(distances_v)
        distances = distances_p + distances_v
        index = np.argmin(distances)
        self.offset = (2 * np.pi * index) / len(self.baseline_pos_loop)