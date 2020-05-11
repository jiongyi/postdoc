from numpy import zeros, hstack, vstack, array, copy, sin, cos, pi, mod, argpartition, logical_and, exp, log, arctan, sign, abs, linspace
from numpy.random import rand, randn, choice, poisson
from numba import jit
from matplotlib.pyplot import subplots

@jit(nopython=True)
def numba_nonzero(row):
    return row.nonzero()

class Network(object):
    def __init__(self, branching_rate_const = 1.0, capping_rate_const = 1.0, total_time = 20.0):
        # Copy argument values.
        self.elongation_rate_const = 141.0 # in nm/ms
        self.branching_rate_const = branching_rate_const
        self.capping_rate_const = capping_rate_const
        self.time_interval = 1e-3
        self.total_time = total_time
        
        # Define constants.
        self.monomer_width = 2.7 # in nm
        self.branching_region_width = 5 * self.monomer_width
        
        # Initialize variables.
        self.pointed_position_mat = zeros((200, 2))
        self.pointed_position_mat[:, 0] = -self.branching_region_width * rand(200)
        self.pointed_position_mat[:, 1] = rand(200) - 0.5
        self.barbed_position_mat = copy(self.pointed_position_mat)
        self.filament_orientation_row = pi * rand(200) - 0.5 * pi
        self.is_capped_row = zeros(200, dtype = bool)
        self.leading_edge_position = 0.0
        self.current_time = 0.0
                     
    def elongate(self, filament_index):
        self.barbed_position_mat[filament_index, 0] += self.monomer_width * cos(self.filament_orientation_row[filament_index])
        self.barbed_position_mat[filament_index, 1] += self.monomer_width * sin(self.filament_orientation_row[filament_index])
        self.barbed_position_mat[filament_index, 1] = mod(self.barbed_position_mat[filament_index, 1], -0.5 * sign(self.barbed_position_mat[filament_index, 1]))
    
    def branch(self, filament_index):
        random_theta = pi * (70 + 5 * randn()) / 180
        if rand() < 0.5:
            random_theta *= -1
        u = cos(self.filament_orientation_row[filament_index])
        v = sin(self.filament_orientation_row[filament_index])
        u_new = u * cos(random_theta) - v * sin(random_theta)
        # Force direction to be towards the leading edge.
        if u_new > 0:
            v_new = u * sin(random_theta) + v * cos(random_theta)
        else:
            u_new = u * cos(random_theta) + v * sin(random_theta)
            v_new = -u * sin(random_theta) + v *  cos(random_theta)
        # Add new filament to arrays.
        self.pointed_position_mat  = vstack((self.pointed_position_mat, array(self.barbed_position_mat[filament_index, :])))
        self.barbed_position_mat = vstack((self.barbed_position_mat, array(self.barbed_position_mat[filament_index, :])))
        self.filament_orientation_row = hstack((self.filament_orientation_row, arctan(v_new / u_new)))
        self.is_capped_row = hstack((self.is_capped_row, False))
        
    def cap(self, filament_index):
        self.is_capped_row[filament_index] = True
        
    def compute_transition_rates(self):
        self.transition_rate_mat = zeros((self.barbed_position_mat.shape[0], 3))
        # Elongation.
        self.transition_rate_mat[~self.active_in_region_index, 0] = self.elongation_rate_const * exp(-0.3 * self.monomer_width / 4.114 / self.active_in_region_index.size * 1e3)
        # Branching is zeroth-order w.r.t. filament count.
        self.transition_rate_mat[self.active_in_region_index, 1] = self.branching_rate_const
        # Capping is first-order w.r.t. filament count.
        self.transition_rate_mat[self.active_in_region_index, 2] = self.capping_rate_const # First order
        
    def catalog_filaments(self):
        self.active_in_region_index = logical_and(self.leading_edge_position - self.barbed_position_mat[:, 0] <= self.branching_region_width, ~self.is_capped_row).nonzero()[0]
        
    def metropolis_step(self):
        for index in self.active_in_region_index:
            # Capping
            if poisson(self.capping_rate_const * self.time_interval) == True:
                self.cap(index)
                continue
            # Branching
            if poisson(self.branching_rate_const * self.time_interval) == True:
                self.branch(index)
            # Elongation
            if poisson(self.elongation_rate_const * self.time_interval) == True:
                self.elongate(index)
        self.current_time += self.time_interval
        self.leading_edge_position += self.elongation_rate_const * self.time_interval
            
        
    def gillespie_step(self):
        nonzero_transition_row_mat, nonzero_transition_col_mat = numba_nonzero(self.transition_rate_mat)
        nonzero_transition_rate_row = self.transition_rate_mat[nonzero_transition_row_mat, nonzero_transition_col_mat].flatten()
        random_mat = rand(nonzero_transition_rate_row.size)
        time_mat = -log(random_mat) / nonzero_transition_rate_row
        random_transition_index = time_mat.argmin()
        time_increment = time_mat[random_transition_index]
        transition_row = nonzero_transition_row_mat[random_transition_index]
        transition_col = nonzero_transition_col_mat[random_transition_index]
        if transition_col == 0:
            self.elongate(transition_row)
            self.leading_edge_position = self.barbed_position_mat[:, 0].max()
        elif transition_col == 1:
            self.branch(transition_row)
        elif transition_col == 2:
            self.cap(transition_row)
        self.current_time += time_increment
                
    def simulate(self):
        self.catalog_filaments()
        while logical_and(self.current_time <= self.total_time, self.active_in_region_index.size != 0):
            self.compute_transition_rates()
            self.metropolis_step()
            #self.gillespie_step()
            self.catalog_filaments()
            
    def compute_order_parameter(self):
        orientation_degree_row = 180 * self.filament_orientation_row[self.active_in_region_index] / pi
        zero_bin_count = (abs(orientation_degree_row) <= 17.5).sum()
        thirtyfive_bin_count = 0.5 * ((abs(orientation_degree_row - 35.0) <= 17.5).sum() + (abs(orientation_degree_row + 35.0) <= 17.5).sum())
        self.order_parameter = (zero_bin_count - thirtyfive_bin_count) / (zero_bin_count + thirtyfive_bin_count)
        
    def display(self):
        self.catalog_filaments()
        fig_hand, axes_hand = subplots()
        axes_hand.plot([self.leading_edge_position, self.leading_edge_position], [-0.5, 0.5], 'black')
        for filament_index in range(self.barbed_position_mat.shape[0]):
            if self.is_capped_row[filament_index] == True:
                axes_hand.plot([self.barbed_position_mat[filament_index, 0], self.pointed_position_mat[filament_index, 0]], 
                              [self.barbed_position_mat[filament_index, 1], self.pointed_position_mat[filament_index, 1]], 'red')
            elif self.is_capped_row[filament_index] == False:
                axes_hand.plot([self.barbed_position_mat[filament_index, 0], self.pointed_position_mat[filament_index, 0]], 
                              [self.barbed_position_mat[filament_index, 1], self.pointed_position_mat[filament_index, 1]], 'blue')
        axes_hand.set_xlabel('x (nm)')
        return fig_hand, axes_hand
    
    def plot_orientation(self):
        fig_hand, axes_hand = subplots()
        axes_hand.hist(180 * self.filament_orientation_row[self.active_in_region_index] / pi, [-90, -87.5, -52.5, -17.5, 17.5, 52.5, 87.5, 90])
        return fig_hand, axes_hand

        