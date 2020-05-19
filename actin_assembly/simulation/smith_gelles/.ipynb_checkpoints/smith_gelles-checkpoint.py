from numpy import zeros, full, hstack, vstack, array, copy, sin, cos, pi, mod, argpartition, logical_and, exp, log, arctan, sign, abs, linspace
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
        self.arp23_loading_rate = 11 * arp23_conc * (42.0 / 220.0)**(1/3)
        self.arp23_unloading_rate = 10.0
        self.arp23_untethering_rate = 1.0 # 8-s lifetime on WCA from three-color paper.
        
        # Define constants.
        self.monomer_width = 2.7 # in nm
        
        # Initialize variables.
        self.num_npfs = int(npf_density)
        self.npf_position_mat = rand(self.num_npfs, 2)
        self.npf_position_mat[:, 0] = 0.0
        self.npf_position_mat[:, 1] -= 0.5
        self.npf_state_mat = zeros((self.num_npfs, 3))
        
        self.barbed_position_mat = rand(200, 2)
        self.barbed_position_mat[:, 0] *= self.monomer_width
        self.barbed_position_mat[:, 1] -= 0.5
        self.filament_orientation_row = pi * rand(200) - 0.5 * pi
        self.is_capped_row = zeros(200, dtype = bool)
        self.is_tethered_row = zeros(200, dtype = bool)
        self.barbed2npf_index = full(200, -1)
        
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
        self.is_tethered_row = hstack((self.is_tethered_row, False))
        self.barbed2end_index = hstack((self.barbed2end_index, -1))
    
    def cap(self, filament_index):
        self.is_capped_row[filament_index] = True
        
    def tether_network(self):
        distance_mat = cdist(self.barbed_position_mat, self.npf_position_mat)
        tether_barbed_index, tether_npf_index = (distance_mat <= self.monomer_width).nonzero()
        for i in range(tether_barbed_index.size):
            if self.is_tethered_row[tether_barbed_index[i]] == False and self.npf_state_mat[tether_npf_index[i],0] == 0 or self.npf_state_mat[tether_npf_index[i], 1] == 1:
                self.is_tethered_row[tether_barbed_index[i]] == True
                self.npf_state_mat[]