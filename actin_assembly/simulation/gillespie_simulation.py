from numpy import pi, sin, cos, hstack, vstack, sign, sqrt, sum, zeros, ones, array, log, cumsum, histogram, reshape, min, pad
from scipy.spatial.distance import cdist
from numpy.random import rand, randn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure

# Define network class.
class network(object):
    def __init__(self, 
                 actin_conc = 5.0, 
                 arp23_conc = 50.0e-3, 
                 cp_conc = 200.0e-3, 
                 npf_density = 1000.0, 
                 total_time = 20.0):
        
        # Calculate rate constants.
        self.elongation_rate = 11 * actin_conc
        self.capping_rate = (42.0 / 63.0)**(1/3) * 11 * cp_conc
        self.actin_loading_rate = 5.5 * actin_conc * (42.0 / (42.0 + 11.0))**(1/3)
        self.actin_transfer_rate = 1.0 # PPR to WH2. Have no idea how fast this should be.
        self.arp23_loading_rate = 11 * arp23_conc * (42.0 / 220.0)**(1/3)
                
        # Declare constants.
        self.square_length = 1.0 # in microns
        self.monomer_length = 2.7e-3 # in microns.
        self.mu_theta = 70 / 180 * pi # mean angle at branches, in radians
        self.mu_sigma = 5 / 180 * pi # variance of angle at branches, in radians
        
        self.bond_stiffness = 1.0 * 10**3
        self.tether_force_scale = 10.0
        
        self.current_time = 0.0 # in seconds
        self.total_time = total_time # Copy argument value.
        
        self.actin_unloading_rate = 3.0 # Based on Zalevsky's paper.
        self.actin_diff_coeff = 1.0 # in square microns per second. 3-30 according to bionumbers.

        # Initialize ends.
        self.no_ends = 200
        self.end_position_mat = rand(self.no_ends, 3)
        self.end_position_mat[:, 0] -= 0.5
        self.end_position_mat[:, 1] -= 0.5
        self.end_position_mat[:, 2] *= self.monomer_length
        self.end_position_mat[:, 2] += self.monomer_length
        azi_angle_col = 2 * pi * rand(self.no_ends, 1)
        polar_angle_col = 0.5 * pi * (1 + rand(self.no_ends, 1))
        self.end_orientation_mat = hstack((sin(polar_angle_col) * cos(azi_angle_col), 
                                           sin(polar_angle_col) * sin(azi_angle_col), 
                                           cos(polar_angle_col)))
        self.is_capped_row = zeros(self.no_ends, dtype = bool)
        
        # Initialize NPFs.
        self.no_npfs = int(npf_density)
        self.npf_position_mat = rand(self.no_npfs, 3)
        self.npf_position_mat[:, 0] -= 0.5
        self.npf_position_mat[:, 1] -= 0.5
        self.npf_position_mat[:, 2] = 0.0
        self.npf_state_mat = zeros((self.no_npfs, 3))
                
    def elongate(self, index):
        self.end_position_mat[index] = self.end_position_mat[index] + self.monomer_length * self.end_orientation_mat[index]
        # Enforce periodic boundary conditions.
        if abs(self.end_position_mat[index, 0]) > 0.5 * self.square_length:
            self.end_position_mat[index, 0] -= sign(self.end_position_mat[index, 0]) * self.square_length
        if abs(self.end_position_mat[index, 1]) > 0.5 * self.square_length:
            self.end_position_mat[index, 1] -= sign(self.end_position_mat[index, 1]) * self.square_length
                 
    def branch(self, index):
        def rotation_angle_axis(ux_axis, uy_axis, uz_axis, theta_axis):
            R_11 = cos(theta_axis) + ux_axis**2 * (1 - cos(theta_axis))
            R_12 = ux_axis * uy_axis * (1 - cos(theta_axis)) - uz_axis * sin(theta_axis)
            R_13 = ux_axis * uz_axis * (1 - cos(theta_axis)) + uy_axis * sin(theta_axis)
            R_21 = uy_axis * ux_axis * (1 - cos(theta_axis)) + uz_axis * sin(theta_axis)
            R_22 = cos(theta_axis) + uy_axis**2 * (1 - cos(theta_axis))
            R_23 = uy_axis * uz_axis * (1 - cos(theta_axis)) - ux_axis * sin(theta_axis)
            R_31 = uz_axis * ux_axis * (1 - cos(theta_axis)) - uy_axis * sin(theta_axis)
            R_32 = uz_axis * uy_axis * (1 - cos(theta_axis)) + ux_axis * sin(theta_axis)
            R_33 = cos(theta_axis) + uz_axis**2 * (1 - cos(theta_axis))
            rotation_mat = array([[R_11, R_12, R_13], [R_21, R_22, R_23], [R_31, R_32, R_33]])
            return rotation_mat
        
        ux_old, uy_old, uz_old = self.end_position_mat[index]
        
        # Find an axis perpendicular to orientation of ended end.
        u_perp_mag = sqrt(2 + (ux_old + uy_old)**2)
        ux_perp_old = 1.0 / u_perp_mag
        uy_perp_old = 1.0 / u_perp_mag
        uz_perp_old = -(ux_old + uy_old) / u_perp_mag
        
        # Perform rotation to find new orientation.
        theta_polar = self.mu_theta + self.mu_sigma * randn()
        theta_azi = 2 * pi * rand()
        polar_rotation_mat = rotation_angle_axis(ux_perp_old, uy_perp_old, uz_perp_old, theta_polar)
        u_new_polar_row = polar_rotation_mat @ array([ux_old, uy_old, uz_old])
        azi_rotation_mat = rotation_angle_axis(ux_old, uy_old, uz_old, theta_azi)
        u_new_row = azi_rotation_mat @ u_new_polar_row
        
        # Do it until it's facing the right way (-z).
        while u_new_row[2] >= 0.0:
            theta_polar = self.mu_theta + self.mu_sigma * randn()
            theta_azi = 2 * pi * rand()
            polar_rotation_mat = rotation_angle_axis(ux_perp_old, uy_perp_old, uz_perp_old, theta_polar)
            u_new_polar_row = polar_rotation_mat @ array([ux_old, uy_old, uz_old])
            azi_rotation_mat = rotation_angle_axis(ux_old, uy_old, uz_old, theta_azi)
            u_new_row = azi_rotation_mat @ u_new_polar_row
                        
        # Add new ended end to relevant arrays.
        self.end_position_mat = vstack((self.end_position_mat, self.end_position_mat[index]))
        self.end_orientation_mat = vstack((self.end_orientation_mat, u_new_row))
        self.is_capped_row = hstack((self.is_capped_row, False))
        self.no_ends += 1
            
    def cap(self, index):
        self.is_capped_row[index] = True
        
    def calculate_transition_rates(self):
        # End transition rate matrix.
        no_npf_monomers = sum(self.npf_state_mat[:, 0] == 1)
        end_transition_rate_mat = zeros((self.no_ends, 3))
        # Elongation.
        end_transition_rate_mat[:, 0] = self.elongation_rate
        end_transition_rate_mat[self.is_capped_row, 0] = 0.0
        distance_end2npf_mat = cdist(self.end_position_mat, self.npf_position_mat[self.npf_state_mat[:, 0] == 1])
        end2npf_elongation_rate_mat = (distance_end2npf_mat**2 / 6 / self.actin_diff_coeff + 1 / self.actin_unloading_rate)**(-1)
        end2npf_elongation_rate_mat[self.is_capped_row, :] = 0
        end_transition_rate_mat = hstack((end_transition_rate_mat, end2npf_elongation_rate_mat))
        # Branching.
        end_transition_rate_mat[:, 1] = self.arp23_loading_rate
        end_transition_rate_mat[self.end_position_mat[:, 2] > self.monomer_length, 1] = 0.0
        # Capping.
        end_transition_rate_mat[:, 2] = self.capping_rate
        
        # npf transition rate matrix.
        npf_transition_rate_mat = zeros((self.no_npfs, 2))
        # Actin loading.
        npf_transition_rate_mat[self.npf_state_mat[:, 0] == 0, 0] = 1.0
        # Actin unloading.
        npf_transition_rate_mat[self.npf_state_mat[:, 0] == 1, 1] = self.actin_unloading_rate
        npf_transition_rate_mat = pad(npf_transition_rate_mat, ((0, 0), (0, end_transition_rate_mat.shape[1] - 2)))
        # Stack transition rate matrices.
        self.transition_rate_mat = vstack((end_transition_rate_mat, npf_transition_rate_mat))
        
    def gillespie_step(self):
        self.calculate_transition_rates()
        total_rate = sum(self.transition_rate_mat)
        time_interval = -log(rand()) / total_rate
        random_rate = total_rate * rand()
        # Switch to nonzero space.
        nonzero_transition_rate_row_row, nonzero_transition_rate_col_row = self.transition_rate_mat.nonzero()
        nonzero_transition_rate_row = self.transition_rate_mat[nonzero_transition_rate_row_row, nonzero_transition_rate_col_row]
        cum_transition_rate_row = cumsum(hstack((0, nonzero_transition_rate_row)))
        count_row, _ = histogram(random_rate, cum_transition_rate_row)
        random_transition_nonzero_index = count_row.nonzero()[0][0] # Only one element.
        index_row = nonzero_transition_rate_row_row[random_transition_nonzero_index]
        index_col = nonzero_transition_rate_col_row[random_transition_nonzero_index]
        if index_row < self.no_ends:
            if index_col == 0:
                self.elongate(index_row)
                self.no_monomers_sol += 1
            elif index_col == 1:
                self.branch(index_row)
            elif index_col == 2:
                self.cap(index_row)
            elif index_col > 2:
                self.elongate(index_row)
                self.no_monomers_npf += 1
                self.npf_state_mat[self.npf_state_mat[:, 0].nonzero()[0][index_col - 3], 0] = 0
        else:
            if index_col == 0:
                self.npf_state_mat[index_row - self.no_ends, 0] = 1
                
            elif index_col == 1:
                self.npf_state_mat[index_row - self.no_ends, 0] = 0             
        self.current_time += time_interval
        min_end_z = min(self.end_position_mat[:, 2])
        if min_end_z < 0:
            self.end_position_mat[:, 2] -= min_end_z
        
    def simulate(self):
        self.no_monomers_sol = 0
        self.no_monomers_npf = 0
        while (self.current_time <= self.total_time) and (sum(~self.is_capped_row) >= 1):
            self.gillespie_step()
        self.network_height = max(self.end_position_mat[:, 2])
        self.normalized_network_growth_rate = self.network_height / (self.monomer_length * self.elongation_rate * self.current_time)
        print(self.normalized_network_growth_rate)
    
    def display_network(self):
        fig1_hand = figure()
        axes1_hand = fig1_hand.add_subplot(111, projection = '3d')
        axes1_hand.quiver(self.end_position_mat[:, 0], self.end_position_mat[:, 1], self.end_position_mat[:, 2], 
                          self.end_orientation_mat[:, 0], self.end_orientation_mat[:, 1], self.end_orientation_mat[:, 2], length = self.elongation_rate * self.monomer_length / self.capping_rate)