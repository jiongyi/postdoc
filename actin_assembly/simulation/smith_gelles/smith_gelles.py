from numpy import zeros, full, hstack, vstack, array, copy, sin, cos, pi, mod, argpartition, logical_and, logical_or, exp, log, arctan, sign, abs, linspace, meshgrid
from numpy.random import rand, randn, choice, poisson
from scipy.spatial.distance import cdist
from numba import jit
from matplotlib.pyplot import subplots

@jit(nopython=True)
def numba_nonzero(row):
    return row.nonzero()

class Network(object):
    def __init__(self, 
                 actin_conc = 5.0, 
                 arp23_conc = 50e-3, 
                 cp_conc = 200e-3, 
                 npf_density = 1000.0, 
                 total_time = 20.0):
        # Calculate and set rate constants.
        self.elongation_rate = 11 * actin_conc
        self.capping_rate = (42.0 / 63.0)**(1/3) * 11 * cp_conc
        self.actin_loading_rate = 5.5 * actin_conc * (42.0 / (42.0 + 11.0))**(1/3)
        self.actin_unloading_rate = 3.0
        self.actin_diff_coeff = 1.0
        self.arp23_loading_rate = 11 * arp23_conc * (42.0 / 220.0)**(1/3)
        self.arp23_unloading_rate = 10.0
        self.arp23_untethering_rate = 1.0 # 8-s lifetime on WCA from three-color paper.
        
        # Define constants.
        self.monomer_width = 2.7e-3 # in nm
        
        # Initialize variables.
        self.num_npfs = int(npf_density)
        self.npf_position_mat = rand(self.num_npfs, 2)
        self.npf_position_mat[:, 0] = 0.0
        self.npf_position_mat[:, 1] -= 0.5
        self.npf_state_mat = zeros((self.num_npfs, 3))
        
        self.num_ends = 200
        self.barbed_position_mat = rand(self.num_ends, 2)
        self.barbed_position_mat[:, 0] *= self.monomer_width
        self.barbed_position_mat[:, 1] -= 0.5
        self.filament_orientation_row = pi * rand(self.num_ends) - 0.5 * pi
        self.is_capped_row = zeros(self.num_ends, dtype = bool)
        self.is_tethered2wh2_row = zeros(self.num_ends, dtype = bool)
        self.is_tethered2arp23_row = zeros(self.num_ends, dtype=bool)
        self.barbed2npf_index = full(self.num_ends, -1)
        
        self.leading_edge_position = 0.0
        self.current_time = 0.0
        self.total_time = total_time
                     
    def elongate(self, filament_index):
        self.barbed_position_mat[filament_index, 0] += self.monomer_width * cos(self.filament_orientation_row[filament_index])
        self.barbed_position_mat[filament_index, 1] += self.monomer_width * sin(self.filament_orientation_row[filament_index])
        self.barbed_position_mat[filament_index, 1] = mod(self.barbed_position_mat[filament_index, 1], 
                                                          -0.5 * sign(self.barbed_position_mat[filament_index, 1]))
    
    def branch(self, filament_index):
        random_theta = pi * (70 + 5 * randn()) / 180
        if rand() < 0.5:
            random_theta *= -1
        u = cos(self.filament_orientation_row[filament_index])
        v = sin(self.filament_orientation_row[filament_index])
        u_new = u * cos(random_theta) - v * sin(random_theta)
        if u_new > 0:
            v_new = u * sin(random_theta) + v * cos(random_theta)
        else:
            u_new = u * cos(random_theta) + v * sin(random_theta)
            v_new = -u * sin(random_theta) + v * cos(random_theta)
        # Add new filament to arrays.
        self.barbed_position_mat = vstack((self.barbed_position_mat, array(self.barbed_position_mat[filament_index, :])))
        self.filament_orientation_row = hstack((self.filament_orientation_row, arctan(v_new / u_new)))
        self.is_capped_row = hstack((self.is_capped_row, False))
        self.is_tethered2wh2_row = hstack((self.is_tethered2wh2_row, False))
        self.is_tethered2arp23_row = hstack((self.is_tethered2arp23_row, False))
        self.barbed2end_index = hstack((self.barbed2end_index, -1))
        self.num_ends += 1
    
    def cap(self, filament_index):
        self.is_capped_row[filament_index] = True
        
    def calculate_transition_rates(self):
        msd_mat = (cdist(self.barbed_position_mat, self.npf_position_mat) + 0.5 * self.monomer_width)**2
        self.transition_rate_mat = zeros((self.num_ends + self.num_npfs, 4 + 2 * self.num_npfs ))
        # Elongation
        can_elongate_row = logical_and(self.is_capped_row == False, logical_or(self.is_tethered2wh2_row == False, self.is_tethered2arp23_row == False))
        self.transition_rate_mat[:self.num_ends, 0][can_elongate_row] = self.elongation_rate
        # WH2 tethering
        is_wh2_empty_row = self.npf_state_mat[:, 0] == 0
        can_tether_row_grid, can_tether_col_grid = meshgrid(can_elongate_row.nonzero()[0], is_wh2_empty_row.nonzero()[0], indexing='ij')
        self.transition_rate_mat[:self.num_ends, 1:(self.num_npfs+1)][can_tether_row_grid, can_tether_col_grid] = (6 * self.actin_diff_coeff / msd_mat[can_tether_row_grid, can_tether_col_grid]) * 1e-3
        # WH2 untethering
        self.transition_rate_mat[:self.num_ends, (self.num_npfs+1)][self.is_tethered2wh2_row] = 0
        # Arp2/3 tethering
        is_ca_loaded_row = self.npf_state_mat[:, 1] == 1
        can_tether2ca_row_grid, can_tether2ca_col_grid = meshgrid((~self.is_tethered2arp23_row).nonzero()[0], is_ca_loaded_row.nonzero()[0], indexing='ij')
        self.transition_rate_mat[:self.num_ends, (self.num_npfs+2) : (2*self.num_npfs+2)][can_tether2ca_row_grid, can_tether2ca_col_grid] = (6 * self.actin_diff_coeff / msd_mat[can_tether2ca_row_grid, can_tether2ca_col_grid]) * 1e-3
        # Arp2/3 untethering
        self.transition_rate_mat[:self.num_ends, 2*self.num_npfs+2][self.is_tethered2arp23_row] = self.arp23_untethering_rate
        # Capping.
        self.transition_rate_mat[:self.num_ends, 2*self.num_npfs+3][can_elongate_row] = self.capping_rate
        # Loading WH2
        self.transition_rate_mat[self.num_ends:, 0][self.npf_state_mat[:, 0] == 0] = self.actin_loading_rate
        # Unloading WH2
        self.transition_rate_mat[self.num_ends:, 1][self.npf_state_mat[:, 0] == 1] = self.actin_unloading_rate
        # Loading CA
        self.transition_rate_mat[self.num_ends:, 2][self.npf_state_mat[:, 1] == 0] = self.arp23_loading_rate
        # Unloading CA
        self.transition_rate_mat[self.num_ends:, 3][self.npf_state_mat[:, 1] == 1] = self.arp23_unloading_rate
        