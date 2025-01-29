import numpy as np
from fun_first_period_estimation import extract_last_period_autocorrelation
from scipy.signal import find_peaks

class Phase_Estimation:
    def __init__(self,
                 step_time                 = 0.01,
                 look_behind_pcent  = 0,
                 look_ahead_pcent = 25,
                 wait_time                 = 5,
                 listening_time            = 15,
                 min_duration_period       = None,
                 reference                 = None,
                 point_1                   = None,
                 point_2                   = None,
                 point_3                   = None):
        
        self.x_vec  = []
        self.y_vec  = []
        self.z_vec  = []
        self.vel_x_vec = []
        self.vel_y_vec = []
        self.vel_z_vec = []
        self.time_vec  = []

        self.step_time                             = step_time      # [s]
        self.initial_time                                = 0         # start time instant of fase computation [s]
        self.wait_time                             = wait_time         # initial waiting time interval [s]
        self.listening_time                        = listening_time        # time interval in witch to estimate the first period [s]
        self.min_duration_period                   = 1         # minimum time duration of periods [s]
        self.look_ahead_pcent  = look_ahead_pcent        # % of the last completed period before the last nearest point on which estimate the new phase
        self.look_behind_pcent   = look_behind_pcent         # % of the last completed period after the last nearest point on which estimate the new phase
        self.diff_len_new_ref                      = 30        # difference in length of the new reference (vector length) compared to the old one, expressed as a percentage of the old length, is accepted.

        self.max_length_vec_period  = 1000
        self.is_first_period_estimated = False
        self.epsilon                = np.pi
        self.offset                 = 0
        self.phase                  = np.zeros(2)
        self.last_completed_period  = np.zeros((self.max_length_vec_period, 6))  
        self.new_period             = np.zeros((self.max_length_vec_period, 6))

        # parameter for the zero phase estimation 
        self.reference = None
        self.point_1   = None
        self.point_2   = None
        self.point_3   = None

        self.counter       = 0
        self.min_index_pre = 0
        self.position_prev = [0, 0, 0]

        self.len_last_period_discarded = 0
        self.range_post                = 0
        self.range_pre                 = 0

        if min_duration_period is not None:
            self.min_duration_period = min_duration_period
        if reference is not None:
            self.reference=reference.copy()
            if point_1 is not None and point_2 is not None and point_3 is not None:
                self.point_1 = point_1
                self.point_2 = point_2
                self.point_3 = point_3
            else:
                raise ValueError("left_chest, right_chest and belly must be different from None ")


    def set_position(self, position, curr_time):  # TODO change name of the function?
        if not self.time_vec:  self.initial_time = curr_time  # initialize initial_time

        self.time_vec.append(curr_time - self.initial_time)

        if  self.time_vec[-1] > self.wait_time:
            self.x_vec.append(position[0])         # TODO should be a numpy matrix
            self.y_vec.append(position[1])
            self.z_vec.append(position[2])
            self.step_time = self.time_vec[-1] - self.time_vec[-2]
            self.vel_x_vec.append((position[0] - self.position_prev[0]) / self.step_time)
            self.vel_y_vec.append((position[1] - self.position_prev[1]) / self.step_time)
            self.vel_z_vec.append((position[2] - self.position_prev[2]) / self.step_time)

        self.position_prev = position
                
        if not self.is_first_period_estimated:
            if self.time_vec[-1] > self.wait_time + self.listening_time:
                self.last_completed_period = extract_last_period_autocorrelation(
                    x_vec = self.x_vec,
                    y_vec = self.y_vec,
                    z_vec = self.z_vec,
                    vel_x_vec = self.vel_x_vec,
                    vel_y_vec = self.vel_y_vec,
                    vel_z_vec = self.vel_z_vec,
                    time_vec = self.time_vec,
                    min_duration_period = self.min_duration_period)

                self.is_first_period_estimated = True
                self.range_post = int(len(self.last_completed_period) * self.look_ahead_pcent / 100)   # range_post is the number of points after the last nearest point on which estimate the new phase
                if self.range_post == 0:
                    self.range_post = 1
                self.range_pre = int(len(self.last_completed_period) * self.look_behind_pcent / 100)
                if self.range_pre == 0:
                    self.range_pre = 1
                if self.reference is not None:
                    self.compute_phase_offset()
                for i in range(len(self.last_completed_period), len(self.x_vec) - 1):
                    p = np.array([self.x_vec[i], self.y_vec[i], self.z_vec[i], self.vel_x_vec[i], self.vel_y_vec[i], self.vel_z_vec[i]])
                    self.phase_computer(p)
                    self.period_estimation(p)

        if len(self.x_vec) > 0:
            p = np.array([self.x_vec[-1], self.y_vec[-1], self.z_vec[-1], self.vel_x_vec[-1], self.vel_y_vec[-1], self.vel_z_vec[-1]])
        else:
            p = np.array([0,0,0,0,0,0])

        self.phase_computer(p) # compute the phase

        if self.is_first_period_estimated == True: # update the completed period estimation
            self.period_estimation(p)

        return np.mod(self.phase[1]+self.offset, 2*np.pi)


    def period_estimation(self,p): # receives position and velocity as input and updates the vector of the last complete period when the phase completes a full cycle.
        if self.phase[1]-self.phase[0] < -self.epsilon:
            if abs(self.counter-len(self.last_completed_period)) < len(self.last_completed_period)*self.diff_len_new_ref/100: # Ensure that the difference in length between the new period and the previous one is not greater than the range set by the user.
                self.last_completed_period = self.new_period[0:self.counter,:]
                self.range_post = int(len(self.last_completed_period) * self.look_ahead_pcent / 100)
                if self.range_post==0:
                    self.range_post=1
                self.range_pre = int(len(self.last_completed_period) * self.look_behind_pcent / 100)
                if self.range_pre == 0:
                    self.range_pre = 1

            elif abs(self.counter-self.len_last_period_discarded) < self.len_last_period_discarded*self.diff_len_new_ref/100:
                self.last_completed_period = self.new_period[0:self.counter,:]
                self.range_post = int(len(self.last_completed_period) * self.look_ahead_pcent / 100)
                if self.range_post==0:
                    self.range_post=1
                self.range_pre = int(len(self.last_completed_period) * self.look_behind_pcent / 100)
                if self.range_pre == 0:
                    self.range_pre = 1
                self.len_last_period_discarded = 0

            else:
                self.len_last_period_discarded = self.counter

            self.new_period = np.zeros((self.max_length_vec_period, 6))
            self.counter = 0
        if(self.counter < self.max_length_vec_period):
            self.new_period[self.counter,:] = p
            self.counter += 1


    def phase_computer(self,p):
        if self.is_first_period_estimated:
            length_reference = len(self.last_completed_period)
            if self.min_index_pre - self.range_pre < 1:
                ref_part1 = self.last_completed_period[0:self.min_index_pre + self.range_post]
                ref_part2 = self.last_completed_period[length_reference - self.range_pre + self.min_index_pre:length_reference]
                ref = np.vstack((ref_part1, ref_part2))

                index_part1 = np.arange(1, self.min_index_pre + self.range_post + 1)
                index_part2 = np.arange(length_reference - self.range_pre + self.min_index_pre, length_reference + 1)
                index = np.concatenate((index_part1, index_part2))
            elif self.min_index_pre + self.range_post > length_reference:
                ref_part1 = self.last_completed_period[self.min_index_pre - self.range_pre:length_reference]
                ref_part2 = self.last_completed_period[0:self.min_index_pre + self.range_post - length_reference]
                ref = np.vstack((ref_part1, ref_part2))

                index_part1 = np.arange(self.min_index_pre - self.range_pre, length_reference + 1)
                index_part2 = np.arange(1, self.min_index_pre + self.range_post - length_reference + 1)
                index = np.concatenate((index_part1, index_part2))
            else:
                ref = self.last_completed_period[self.min_index_pre - self.range_pre:self.min_index_pre + self.range_post]
                index = np.arange(self.min_index_pre - self.range_pre, self.min_index_pre + self.range_post + 1)

            ref_p=ref.copy()
            ref_v=ref.copy()
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
            self.phase[0] = self.phase[1]
            self.phase[1] = (2 * np.pi  * self.min_index_pre)/ length_reference
            if self.phase[1]-self.phase[0] > self.epsilon:  # Avoid 0 to 2pi jumps
                 self.phase[1] = self.phase[0]
            if self.phase[1]-self.phase[0] < 0 and self.phase[0]-self.phase[1] < self.epsilon:
                self.phase[1] = self.phase[0]
        
        else:
            self.phase[0] = 0
            self.phase[1] = 0

    
    def compute_phase_offset(self):
        x_axis = (self.point_2 - self.point_1) / np.linalg.norm(self.point_2 - self.point_1)
        z_vector = self.point_3 - self.point_2
        z_axis = z_vector - np.dot(z_vector, x_axis) * x_axis 
        z_axis = z_axis / np.linalg.norm(z_axis)  
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis])
        
        rotated_ref = np.dot(self.last_completed_period[:, 0:3], rotation_matrix.T)
        
        centroid = np.mean(rotated_ref, axis=0)
        rotated_centered_ref = rotated_ref - centroid

        scale_factors = np.std(self.reference, axis=0) / np.std(rotated_centered_ref, axis=0)
        scale_factors[np.isnan(scale_factors)] = 1 
        scaled_rotated_centered_ref = rotated_centered_ref * scale_factors

        p = scaled_rotated_centered_ref[0, :]
        vel=(scaled_rotated_centered_ref[1, :]-scaled_rotated_centered_ref[0, :])/self.step_time
        vel_ref=np.gradient(self.reference, np.arange(0, len(self.reference)*0.01, 0.01), axis=0)
        position_norm = self.reference.copy()
        vel_norm = vel_ref.copy()
        for i in range(len(self.reference)):
            position_norm[i] = (self.reference[i] - p) ** 2
            vel_norm[i] = (vel_ref[i] - vel) ** 2
        distances_p = np.sqrt(np.sum((position_norm), axis=1))
        distances_v = np.sqrt(np.sum((vel_norm), axis=1))
        distances_p = distances_p/max(distances_p)
        distances_v = distances_v/max(distances_v)
        distances = distances_p + distances_v
        index = np.argmin(distances)
        self.offset = (2 * np.pi * index) / len(self.reference)
        
