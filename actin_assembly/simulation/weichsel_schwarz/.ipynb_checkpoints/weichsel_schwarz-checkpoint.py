from numpy import zeros, hstack, vstack, array, copy, sin, cos, pi, mod, argpartition, logical_and, intersect1d, log, arctan, sign, abs, linspace
from numpy.random import rand, randn, choice, poisson
from numba import jit
from matplotlib.pyplot import subplots

@jit(nopython=True)
def numba_nonzero(row):
    return row.nonzero()

class Network(object):
    def __init__(self, branching_rate_const = 30.0, capping_rate_const = 0.05, num_front_filaments = 2, total_time = 20.0):
        # Copy argument values.
        self.elongation_rate_const = 370.0 # in nm/ms
        self.branching_rate_const = branching_rate_const
        self.capping_rate_const = capping_rate_const
        self.num_front_filaments = num_front_filaments
        self.total_time = total_time
        
        # Define constants.
        self.monomer_width = 1.0 # in nm
        self.branching_region_width = 2 * self.monomer_width
        
        # Initialize variables.
        self.pointed_position_mat = zeros((150, 2))
        self.pointed_position_mat[:, 0] = -self.branching_region_width * rand(150)
        self.pointed_position_mat[:, 1] = rand(150) - 0.5
        self.barbed_position_mat = copy(self.pointed_position_mat)
        self.filament_orientation_row = pi * rand(150) - 0.5 * pi
        self.is_capped_row = zeros(150, dtype = bool)
        self.leading_edge_position = 0.0
        self.current_time = 0.0
                     
    def elongate(self, filament_index):
        self.barbed_position_mat[filament_index, 0] += cos(self.filament_orientation_row[filament_index])
        self.barbed_position_mat[filament_index, 1] += sin(self.filament_orientation_row[filament_index])
        self.barbed_position_mat[filament_index, 1] = mod(self.barbed_position_mat[filament_index, 1], -0.5 * sign(self.barbed_position_mat[filament_index, 1]))
    
    def branch(self, filament_index):
        random_theta = pi * (70 + 5 * randn()) / 180
        if rand() < 0.5:
            random_theta *= -1
        u = cos(self.filament_orientation_row[filament_index])
        v = sin(self.filament_orientation_row[filament_index])
        u_new = u * cos(random_theta) - v * sin(random_theta)
        v_new = u * sin(random_theta) + v * cos(random_theta)
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
        self.transition_rate_mat[~self.is_capped_row, 0] = self.elongation_rate_const
        self.transition_rate_mat[self.front_filament_index, 0] = 0.0
        # Branching is zeroth-order w.r.t. filament count.
        if self.active_filament_index.size > 0:
            self.transition_rate_mat[self.active_filament_index, 1] = self.branching_rate_const / self.active_filament_index.size
        # Capping is first-order w.r.t. filament count.
        self.transition_rate_mat[self.active_uncapped_filament_index, 2] = self.capping_rate_const # First order
        
    def catalog_filaments(self):
        self.front_filament_index = argpartition(self.barbed_position_mat[:, 0], -self.num_front_filaments)[-self.num_front_filaments:]
        self.front_boundary_position = self.barbed_position_mat[self.front_filament_index, 0].min()
        is_filament_behind_row = self.front_boundary_position > self.barbed_position_mat[:, 0]
        is_filament_close_row = self.barbed_position_mat[:, 0] >= (self.front_boundary_position - self.branching_region_width)
        is_filament_active_row = logical_and(is_filament_behind_row, is_filament_close_row)
        is_filament_active_uncapped_row = logical_and(is_filament_active_row, ~self.is_capped_row)
        self.active_filament_index = is_filament_active_row.nonzero()[0]
        self.active_uncapped_filament_index = is_filament_active_uncapped_row.nonzero()[0]
        
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
        elif transition_col == 1:
            self.branch(transition_row)
        elif transition_col == 2:
            self.cap(transition_row)
        self.current_time += time_increment
                
    def simulate(self):
        while logical_and(self.current_time <= self.total_time, (~self.is_capped_row).sum() != 0):
            self.catalog_filaments()
            self.compute_transition_rates()
            self.gillespie_step()
            self.leading_edge_position = self.barbed_position_mat[:, 0].max()
            
    def compute_order_parameter(self):
        orientation_degree_row = 180 * self.filament_orientation_row[self.active_filament_index] / pi
        zero_bin_count = (abs(orientation_degree_row) <= 17.5).sum()
        thirtyfive_bin_count = 0.5 * ((abs(orientation_degree_row - 35.0) <= 17.5).sum() + (abs(orientation_degree_row + 35.0) <= 17.5).sum())
        self.order_parameter = (zero_bin_count - thirtyfive_bin_count) / (zero_bin_count + thirtyfive_bin_count)
        
    def display(self):
        self.catalog_filaments()
        fig_hand, axes_hand = subplots()
        axes_hand.plot([self.leading_edge_position, self.leading_edge_position], [-0.5, 0.5], 'black')
        axes_hand.plot([self.front_boundary_position, self.front_boundary_position], [-0.5, 0.5], 'green')
        for filament_index in range(self.barbed_position_mat.shape[0]):
            if self.is_capped_row[filament_index] == True:
                axes_hand.plot([self.barbed_position_mat[filament_index, 0], self.pointed_position_mat[filament_index, 0]], 
                              [self.barbed_position_mat[filament_index, 1], self.pointed_position_mat[filament_index, 1]], 'red')
            elif self.is_capped_row[filament_index] == False:
                axes_hand.plot([self.barbed_position_mat[filament_index, 0], self.pointed_position_mat[filament_index, 0]], 
                              [self.barbed_position_mat[filament_index, 1], self.pointed_position_mat[filament_index, 1]], 'blue')
        axes_hand.set_xlabel('Monomer lengths')
        return fig_hand, axes_hand
    
    def plot_orientation(self):
        fig_hand, axes_hand = subplots()
        axes_hand.hist(180 * self.filament_orientation_row[self.active_filament_index] / pi, [-90, -87.5, -52.5, -17.5, 17.5, 52.5, 87.5, 90])
        return fig_hand, axes_hand

        