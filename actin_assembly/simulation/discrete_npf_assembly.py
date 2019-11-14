from numpy import pi, zeros, cos, sin, copy, max, flatnonzero, sqrt, nan, argmin, isnan
from numpy.random import rand
class network(object):
    def __init__(self,
                 no_npfs = 50, load_rate_const = 10.0, transfer_rate_const = 1.0, nucleation_rate_const = 0.1, capping_rate_const = 0.1):
        # Copy argument values.
        self.no_npfs = no_npfs
        self.load_rate_const = load_rate_const
        self.transfer_rate_const = transfer_rate_const
        self.nucleation_rate_const = nucleation_rate_const
        self.capping_rate_const = capping_rate_const
        
        # Define constants.
        self.leading_edge_width = 1e3
        self.monomer_width = 2.7
        self.mu_theta = 70.0 / 180.0 * pi
        self.mu_sigma = 5.0 / 180.0 * pi
        self.untether_rate_const = 1.0
        self.thermal_force = 10.0
        self.spring_constant = 1.0
        
        # Initialize filaments.
        self.no_filaments = 200
        self.no_monomers = 0
        self.x_point_row = zeros(self.no_filaments)
        self.y_point_row = self.leading_edge_width * rand(self.no_filaments)
        self.theta_row = pi * rand(self.no_filaments) - 0.5 * pi
        self.u_row = cos(self.theta_row)
        self.v_row = sin(self.theta_row)
        self.x_barb_row = copy(self.x_point_row) + self.u_row
        self.y_barb_row = copy(self.y_point_row) + self.v_row
        self.is_capped_row = zeros(self.no_filaments, dtype = bool)
        self.is_tethered_row = zeros(self.no_filaments, dtype = bool)
        
        # Initialize npfs.
        self.x_npf_row = zeros(self.no_npfs)
        self.y_npf_row = self.leading_edge_width * rand(self.no_npfs)
        self.npf_bound_state_row = zeros((self.no_npfs, 3))
        self.index_nearest_end_row = zeros(self.no_npfs)
        self.index_nearest_end_row.fill(nan)

        # Initialize network parameters.
        self.network_load = 0.0
        self.x_leading = max(self.x_barb_row)
        
        # Initialize simulation parameters.
        self.time = 0.0
        self.time_interval = 1e-3
        
    def update_proximal_end_indices(self):
        distance_row = self.x_leading - self.x_barb_row
        is_proximal_row = distance_row <= self.monomer_width
        self.index_proximal_end_row = flatnonzero(is_proximal_row)
        
    def update_nearest_end_indices(self):
        for i_npf in range(self.no_npfs):
            i_x_npf = self.x_npf_row[i_npf]
            i_y_npf = self.y_npf_row[i_npf]
            i_distance_row = sqrt((self.x_barb_row - i_x_npf)**2 + (self.y_barb_row - i_y_npf)**2)
            self.index_nearest_end_row[i_npf] = argmin(i_distance_row)
            
    def update_step(self):
        for i_npf in range(self.no_npfs):
            i_nearest_end = self.index_nearest_end_row[i_npf]
            if self.npf_bound_state_row[i_npf] = [0, 0, 0]:
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][0] = 1
                if ~isnan(i_nearest_end):
                    self.npf_bound_state_row[i_npf][1] = 2
                    self.is_tethered_row[i_nearest_end] = True
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][2] = 1
            elif self.npf_bound_state_row[i_npf] = [0, 1, 0]:
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][0] = 1
                if ~isnan(i_nearest_end):
                    if ~self.is_capped_row[i_nearest_end]:
                        self.elongate(i_nearest_end)
                        self.npf_bound_state_row[i_npf][1] = 0
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][2] = 1
            elif self.npf_bound_state_row[i_npf] = [0, 2, 0]:
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][0] = 1
                if rand() <= (self.untether_rate_const * self.time_interval):
                    self.is_tethered_row[i_nearest_end] = False
                    self.npf_bound_state_row[i_npf][1] = 0
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][2] = 1
            elif self.npf_bound_state_row[i_npf] = [0, 0, 1]:
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][0] = 1
                if ~isnan(i_nearest_end):
                    if ~self.is_capped_row[i_nearest_end]:
                        self.npf_bound_state_row[i_npf][1] = 2
                        self.is_tethered_row[i_nearest_end] = True
            elif self.npf_bound_state_row[i_npf] = [0, 1, 1]:
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][0] = 1
            elif self.npf_bound_state_row[i_npf] = [0, 2, 1]:
                if rand() <= (self.load_rate_const * self.time_interval):
                    self.npf_bound_state_row[i_npf][0] = 1
                if rand() <= (self.untether_rate_const * self.time_interval):
                    self.is_tethered_row[i_nearest_end] = False
                    self.npf_bound_state_row[i_npf][1] = 0
            elif self.npf_bound_state_row[i_npf] = [1, 0, 0]:
                if ~isnan(i_nearest_end):
                    if 
            elif self.npf_bound_state_row[i_npf] = [1, 1, 0]:
            elif self.npf_bound_state_row[i_npf] = [1, 2, 0]:
            elif self.npf_bound_state_row[i_npf] = [1, 0, 1]:
            elif self.npf_bound_state_row[i_npf] = [1, 1, 1]:
            elif self.npf_bound_state_row[i_npf] = [1, 2, 1]:
        
        
        
        