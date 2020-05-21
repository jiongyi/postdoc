from numba import jit
from numpy import pi, sin, cos, hstack, vstack, sign, sqrt, sum, zeros, ones, array, log, cumsum, reshape, min, pad, full, searchsorted, unravel_index, logical_and, meshgrid
from scipy.spatial.distance import cdist
from numpy.random import rand, randn, choice
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure

# Helper function
@jit(nopython=True)
def nonzero_numba(mat):
    return mat.nonzero()

# Define network class.
class network(object):
    def __init__(self,
                 actin_conc = 5.0,
                 arp23_conc = 50.0e-3,
                 cp_conc = 200.0e-3,
                 npf_density = 1000.0,
                 total_time = 20.0):

        # Calculate and set rate constants.
        self.elongation_rate = 11 * actin_conc
        self.capping_rate = (42.0 / 63.0)**(1/3) * 11 * cp_conc
        self.actin_loading_rate = 5.5 * actin_conc * (42.0 / (42.0 + 11.0))**(1/3)
        self.arp23_loading_rate = 11 * arp23_conc * (42.0 / 220.0)**(1/3)
        self.arp23_unloading_rate = 10.0
        self.arp23_untethering_rate = 1.0 # 8-s lifetime on WCA from three-color paper.

        # Declare constants.
        self.square_length = 1.0 # in microns
        self.monomer_length = 2.7e-3 # in microns.
        self.mu_theta = 70 / 180 * pi # mean angle at branches, in radians
        self.mu_sigma = 5 / 180 * pi # variance of angle at branches, in radians

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
        self.is_tethered_row = zeros(self.no_ends, dtype = bool)
        self.index_end2npf_tether_row = full(self.no_ends, -1)

        # Initialize NPFs.
        self.no_npfs = int(npf_density)
        self.npf_position_mat = rand(self.no_npfs, 3)
        self.npf_position_mat[:, 0] -= 0.5
        self.npf_position_mat[:, 1] -= 0.5
        self.npf_position_mat[:, 2] = 0.0
        self.npf_state_mat = zeros((self.no_npfs, 4), dtype = bool)

        self.no_monomers_npf = 0
        self.no_monomers_sol = 0

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

        # Add new barbed end to relevant arrays.
        self.end_position_mat = vstack((self.end_position_mat, self.end_position_mat[index]))
        self.end_orientation_mat = vstack((self.end_orientation_mat, u_new_row))
        self.is_capped_row = hstack((self.is_capped_row, False))
        self.is_tethered_row = hstack((self.is_tethered_row, False))
        self.index_end2npf_tether_row = hstack((self.index_end2npf_tether_row, -1))
        self.no_ends += 1

    def cap(self, index):
        self.is_capped_row[index] = True

    def tether(self, index_end, index_npf):
        self.is_tethered_row[index_end] = True
        self.npf_state_mat[index_npf, 3] = True
        self.index_end2npf_tether_row[index_end] = index_npf
        # Move barbed end to npf.
        self.end_position_mat[index_end, 0] = self.npf_position_mat[index_npf, 0]
        self.end_position_mat[index_end, 1] = self.npf_position_mat[index_npf, 1]

    def untether(self, index_end):
        index_npf = self.index_end2npf_tether_row[index_end]
        self.is_tethered_row[index_end] = False
        self.npf_state_mat[index_npf, 2] = False # Take Arp2/3 with it.
        self.npf_state_mat[index_npf, 3] = False
        self.index_end2npf_tether_row[index_end] = -1
        if self.npf_state_mat[index_npf, 0] == True:
            self.branch(index_end)
            self.npf_state_mat[index_npf, 0] == False

    def calculate_transition_rates(self):
        msd_mat = (cdist(self.end_position_mat, self.npf_position_mat) + 0.1 * self.monomer_length)**2
        no_ends = self.no_ends
        no_npfs = self.no_npfs
        self.transition_rate_mat = zeros((no_ends + no_npfs, 1 + no_npfs + no_npfs + 1 + 1))
        # Compute elongation rates.
        can_elongate_bool = logical_and(self.is_capped_row == False, self.is_tethered_row == False)
        #actin_npf_rate_mat = (self.actin_unloading_rate**-1 + msd_mat / 6 / self.actin_diff_coeff)**-1
        #actin_npf_rate_mat[:, self.npf_state_mat[:, 0] == False] = False
        self.transition_rate_mat[:no_ends, 0][can_elongate_bool] = self.elongation_rate
        is_loaded_bool = self.npf_state_mat[:, 0] == True
        row_grid, col_grid = meshgrid(can_elongate_bool.nonzero()[0], is_loaded_bool.nonzero()[0], indexing = 'ij')
        self.transition_rate_mat[:no_ends, 1:(no_npfs + 1)][row_grid, col_grid] = (self.actin_unloading_rate**-1 + msd_mat[row_grid, col_grid] / 6 / self.actin_diff_coeff)**-1
        # Arp2/3 tethering.
        can_tether_bool = logical_and(self.npf_state_mat[:, 2], self.npf_state_mat[:, 3] == False)
        self.transition_rate_mat[:no_ends, (no_npfs + 2) : (2 * no_npfs + 2)][:, can_tether_bool] = (6 * self.actin_diff_coeff / msd_mat[:, can_tether_bool]) * 1e-3
        # self.transition_rate_mat[:no_ends, (no_npfs + 2) : (2 * no_npfs + 2)] = arp23_tethering_rate_mat
        # Arp23-untethering rates.
        self.transition_rate_mat[:no_ends, (2 * no_npfs + 1)][self.is_tethered_row == True] = self.arp23_untethering_rate
        # Capping rates.
        self.transition_rate_mat[:no_ends, (2 * no_npfs + 2)][self.is_capped_row == False] = self.capping_rate

        # NPF transition rates.
        # Loading WH2 with actin.
        self.transition_rate_mat[no_ends:, 0][self.npf_state_mat[:, 0] == False] = self.actin_loading_rate
        # Actin unloading from WH2.
        self.transition_rate_mat[no_ends:, 1][self.npf_state_mat[:, 0] == True] = self.actin_unloading_rate
        # Loading CA with Arp2/3
        self.transition_rate_mat[no_ends:, 2][self.npf_state_mat[:, 2] == False] = self.arp23_loading_rate
        # Untethered Arp2/3 unloading from CA.
        self.transition_rate_mat[no_ends:, 3][logical_and(self.npf_state_mat[:, 2] == True, self.npf_state_mat[:, 3] == False)] = self.arp23_unloading_rate

    def gillespie_step(self):
        #Switch to nonzero space.
        nonzero_row_mat, nonzero_col_mat = nonzero_numba(self.transition_rate_mat)
        nonzero_transition_rate_mat = self.transition_rate_mat[nonzero_row_mat, nonzero_col_mat]
        total_rate = nonzero_transition_rate_mat.sum()
        time_interval = -log(rand()) / total_rate
        transition_probability_row = nonzero_transition_rate_mat.flatten() / total_rate
        random_nonzero_transition_index = choice(transition_probability_row.size, p = transition_probability_row)
        #Go back to sparse space.
        index_row = nonzero_row_mat[random_nonzero_transition_index]
        index_col = nonzero_col_mat[random_nonzero_transition_index]
        #index_row, index_col = unravel_index(random_transition_index, self.transition_rate_mat.shape)
        if index_row < self.no_ends:
            bin_edge_row = cumsum([0, 1, self.no_npfs, self.no_npfs, 1, 1])
            index_bin = searchsorted(bin_edge_row > index_col, True) - 1
            if index_bin == 0:
                self.elongate(index_row)
                self.no_monomers_sol += 1
            elif index_bin == 1:
                self.elongate(index_row)
                self.npf_state_mat[index_col - bin_edge_row[index_bin], 0] = False
                self.no_monomers_npf += 1
            elif index_bin == 2:
                self.tether(index_row, index_col - bin_edge_row[index_bin])
            elif index_bin == 3:
                self.untether(index_row)
            elif index_bin == 4:
                self.cap(index_row)
        else:
            if index_col == 0:
                # Load actin
                self.npf_state_mat[index_row - self.no_ends, 0] = True
            elif index_col == 1:
                # Unload actin
                self.npf_state_mat[index_row - self.no_ends, 0] = False
            elif index_col == 2:
                # Load arp2/3 complex.
                self.npf_state_mat[index_row - self.no_ends, 2] = True
            elif index_col == 3:
                # Unload arp2/3 complex.
                self.npf_state_mat[index_row - self.no_ends, 2] = False
        # Update time and space.
        self.current_time += time_interval
        if sum(~self.is_tethered_row) > 0:
            min_end_z = min(self.end_position_mat[~self.is_tethered_row, 2])
            if min_end_z < 0:
                self.end_position_mat[~self.is_tethered_row, 2] -= min_end_z

    def simulate(self):
        i_percent = 0.1
        no_iterations = 0
        while (self.current_time <= self.total_time) and (sum(~self.is_capped_row) >= 1):
            self.calculate_transition_rates()
            self.gillespie_step()
            if self.current_time >= self.total_time * (i_percent):
                print(i_percent * 100)
                i_percent += 0.1
            no_iterations += 1
        self.network_height = max(self.end_position_mat[:, 2])
        self.normalized_network_growth_rate = self.network_height / (self.monomer_length * self.elongation_rate * self.current_time)
        print(no_iterations)
        print(self.normalized_network_growth_rate)

    def display(self):
        arrow_length = 0.1 * self.end_position_mat[:, 2].max()
        fig1_hand = figure()
        axes1_hand = fig1_hand.add_subplot(111, projection = '3d')
        axes1_hand.quiver(self.end_position_mat[:, 0], self.end_position_mat[:, 1], self.end_position_mat[:, 2],
                          self.end_orientation_mat[:, 0], self.end_orientation_mat[:, 1], self.end_orientation_mat[:, 2], length = arrow_length)
        return fig1_hand, axes1_hand
