import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.spatial.distance import cdist


@jit(nopython=True)
def searchsorted_left(a, v):
    y = np.searchsorted(a, v, side='left')
    return y

@jit(nopython=True)
def searchsorted_right(a, v):
    y = np.searchsorted(a, v, side='right')
    return y


class Network(object):
    def __init__(self,
                 actin_umolar=5.0,
                 arp23_umolar=50.0e-3,
                 cp_umolar=50e-3,
                 no_npfs=1000,
                 total_time=1.0):
        # Kinetics
        self.k_monomer_on_wh2 = 5.0 * actin_umolar
        self.k_monomer_off_wh2 = 3.0
        self.k_barbed_off_wh2 = 3.0
        self.k_barbed_monomer_off_wh2 = 10 * self.k_barbed_off_wh2

        self.k_arp23_on_ca = 100.0 * arp23_umolar
        self.k_arp23_off_ca = 1.0e-1
        self.k_barbed_slow_off_arp23_ca = 1.0
        self.k_barbed_fast_off_arp23_ca = 10.0
        self.k_barbed_arp23_off_ca = 0.7

        self.k_elongate = 11.0 * actin_umolar
        self.k_cap = 3.0 * cp_umolar

        self.barbed_diff_coeff = 8000.0e-6
        # Spatial constants.
        self.monomer_length = 2.7e-3
        self.branch_angle_mu = 70 / 180 * np.pi
        self.branch_angle_sigma = 5 / 180 * np.pi

        # Barbed ends.
        self.no_barbed = 200
        self.barbed_xyz_mat = np.random.rand(self.no_barbed, 3)
        self.barbed_xyz_mat[:, [0, 1]] -= 0.5
        self.barbed_xyz_mat[:, 2] *= self.monomer_length

        random_phi_col = 2 * np.pi * np.random.rand(self.no_barbed, 1)
        random_theta_col = 0.5 * np.pi * (1 + np.random.rand(self.no_barbed, 1))
        self.barbed_orientation_mat = np.concatenate((np.sin(random_theta_col) + np.cos(random_phi_col),
                                                      np.sin(random_theta_col) + np.sin(random_phi_col),
                                                      np.cos(random_theta_col)), axis=1)

        self.barbed_is_capped_row = np.zeros(self.no_barbed, dtype=bool)
        self.barbed_has_wh2_row = np.zeros(self.no_barbed, dtype=bool)
        self.barbed_has_monomer_wh2_row = np.zeros(self.no_barbed, dtype=bool)
        self.barbed_has_weak_arp23_ca_row = np.zeros(self.no_barbed, dtype=bool)
        self.barbed_has_strong_arp23_ca_row = np.zeros(self.no_barbed, dtype=bool)
        self.barbed_has_active_arp23_ca_row = np.zeros(self.no_barbed, dtype=bool)
        self.barbed2npf_index_row = np.full(self.no_barbed, -1)

        # Nucleation promoting factors.
        self.no_npfs = no_npfs
        self.npf_xyz_mat = np.random.rand(self.no_npfs, 3)
        self.npf_xyz_mat[:, [0, 1]] -= 0.5
        self.npf_xyz_mat[:, 2] = 0.0
        self.wh2_has_monomer_row = np.zeros(self.no_npfs, dtype=bool)
        self.wh2_has_barbed_row = np.zeros(self.no_npfs, dtype=bool)
        self.wh2_has_monomer_barbed_row = np.zeros(self.no_npfs, dtype=bool)
        self.ca_has_arp23_row = np.zeros(self.no_npfs, dtype=bool)
        self.ca_arp23_has_barbed_row = np.zeros(self.no_npfs, dtype=bool)

        self.transition_rate_mat = np.array([])
        self.transition_rate_edge_row = np.array([])

        # Simulation parameters
        self.current_time = 0.0
        self.total_time = total_time

    def calculate_transition_rates(self):
        # Elongation from solution
        k_elongate_col = self.k_elongate * ~self.barbed_is_capped_row[:, np.newaxis]
        k_elongate_col[self.barbed_has_wh2_row] = 0.0
        k_elongate_col[self.barbed_has_monomer_wh2_row] = 0.0
        k_elongate_col[self.barbed_has_weak_arp23_ca_row] = 0.0
        k_elongate_col[self.barbed_has_strong_arp23_ca_row] = 0.0
        k_elongate_col[self.barbed_has_active_arp23_ca_row] = 0.0
        # Capping from solution
        k_cap_col = self.k_cap * ~self.barbed_is_capped_row[:, np.newaxis]
        k_cap_col[self.barbed_has_wh2_row] = 0.0
        k_cap_col[self.barbed_has_monomer_wh2_row] = 0.0
        k_cap_col[self.barbed_has_weak_arp23_ca_row] = 0.0
        k_cap_col[self.barbed_has_strong_arp23_ca_row] = 0.0
        k_cap_col[self.barbed_has_active_arp23_ca_row] = 0.0
        # Load WH2 domains
        k_monomer_on_wh2_mat = np.zeros((self.no_barbed, self.no_npfs))
        k_monomer_on_wh2_mat[0, ~self.wh2_has_monomer_row] = self.k_monomer_on_wh2
        k_monomer_on_wh2_mat[0, self.wh2_has_monomer_barbed_row] = 0.0
        # Unload WH2 domains
        k_monomer_off_wh2_mat = np.zeros((self.no_barbed, self.no_npfs))
        k_monomer_off_wh2_mat[0, self.wh2_has_monomer_row] = self.k_monomer_off_wh2
        k_monomer_off_wh2_mat[0, self.wh2_has_monomer_barbed_row] = 0.0
        # Tether to empty WH2 domain
        k_barbed_on_wh2_mat = self.barbed_diff_coeff / cdist(self.barbed_xyz_mat, self.npf_xyz_mat) ** 2
        k_barbed_on_wh2_mat[self.barbed_is_capped_row, :] = 0.0
        k_barbed_on_wh2_mat[:, self.wh2_has_monomer_row] = 0.0
        k_barbed_on_wh2_mat[:, self.wh2_has_barbed_row] = 0.0
        k_barbed_on_wh2_mat[:, self.ca_has_arp23_row] = 0.0
        k_barbed_on_wh2_mat[:, self.ca_arp23_has_barbed_row] = 0.0
        # Break tether from WH2 domain
        k_barbed_off_wh2_col = self.k_barbed_off_wh2 * self.barbed_has_wh2_row[:, np.newaxis]
        # Tether to loaded WH2 domain
        k_barbed_on_monomer_wh2_mat = self.barbed_diff_coeff / cdist(self.barbed_xyz_mat, self.npf_xyz_mat) ** 2
        k_barbed_on_monomer_wh2_mat[self.barbed_is_capped_row, :] = 0.0
        k_barbed_on_monomer_wh2_mat[:, ~self.wh2_has_monomer_row] = 0.0
        k_barbed_on_monomer_wh2_mat[:, self.wh2_has_monomer_barbed_row] = 0.0
        k_barbed_on_monomer_wh2_mat[:, self.ca_has_arp23_row] = 0.0
        # Break tether and take monomer from WH2 domain
        k_barbed_off_monomer_wh2_col = self.k_barbed_monomer_off_wh2 * self.barbed_has_monomer_wh2_row[:, np.newaxis]
        # Load CA domain
        k_arp23_on_ca_mat = np.zeros((self.no_barbed, self.no_npfs))
        k_arp23_on_ca_mat[0, ~self.ca_has_arp23_row] = self.k_arp23_on_ca
        # Unload CA domain
        k_arp23_off_ca_mat = np.zeros((self.no_barbed, self.no_npfs))
        k_arp23_off_ca_mat[0, self.ca_has_arp23_row] = self.k_arp23_off_ca
        k_arp23_off_ca_mat[0, self.ca_arp23_has_barbed_row] = 0.0
        # Tether to loaded CA domain
        k_barbed_on_arp23_ca_mat = self.barbed_diff_coeff / cdist(self.barbed_xyz_mat, self.npf_xyz_mat) ** 2
        k_barbed_on_arp23_ca_mat[self.barbed_has_weak_arp23_ca_row, :] = 0.0
        k_barbed_on_arp23_ca_mat[self.barbed_has_strong_arp23_ca_row, :] = 0.0
        k_barbed_on_arp23_ca_mat[self.barbed_has_active_arp23_ca_row, :] = 0.0
        k_barbed_on_arp23_ca_mat[:, ~self.ca_has_arp23_row] = 0.0
        k_barbed_on_arp23_ca_mat[:, ~self.ca_arp23_has_barbed_row] = 0.0
        # Break tether from weakly bound Arp2/3
        k_barbed_off_weak_arp23_ca_col = self.k_barbed_fast_off_arp23_ca * self.barbed_has_weak_arp23_ca_row[:, np.newaxis]
        # Break tether from strongly bound Arp2/3
        k_barbed_off_strong_arp23_ca_col = self.k_barbed_slow_off_arp23_ca * self.barbed_has_strong_arp23_ca_row[:, np.newaxis]
        # Break tether and take activated Arp2/3
        k_barbed_off_active_arp23_ca_col = self.k_barbed_arp23_off_ca * self.barbed_has_active_arp23_ca_row[:, np.newaxis]
        self.transition_rate_mat = np.concatenate((k_elongate_col,
                                                   k_cap_col,
                                                   k_monomer_on_wh2_mat,
                                                   k_monomer_off_wh2_mat,
                                                   k_barbed_on_wh2_mat,
                                                   k_barbed_off_wh2_col,
                                                   k_barbed_on_monomer_wh2_mat,
                                                   k_barbed_off_monomer_wh2_col,
                                                   k_arp23_on_ca_mat,
                                                   k_arp23_off_ca_mat,
                                                   k_barbed_on_arp23_ca_mat,
                                                   k_barbed_off_weak_arp23_ca_col,
                                                   k_barbed_off_strong_arp23_ca_col,
                                                   k_barbed_off_active_arp23_ca_col), axis=1)
        self.transition_rate_edge_row = np.cumsum(np.array([k_elongate_col.shape[1],
                                                            k_cap_col.shape[1],
                                                            k_monomer_on_wh2_mat.shape[1],
                                                            k_monomer_off_wh2_mat.shape[1],
                                                            k_barbed_on_wh2_mat.shape[1],
                                                            k_barbed_off_wh2_col.shape[1],
                                                            k_barbed_on_monomer_wh2_mat.shape[1],
                                                            k_barbed_off_monomer_wh2_col.shape[1],
                                                            k_arp23_on_ca_mat.shape[1],
                                                            k_arp23_off_ca_mat.shape[1],
                                                            k_barbed_on_arp23_ca_mat.shape[1],
                                                            k_barbed_off_weak_arp23_ca_col.shape[1],
                                                            k_barbed_off_strong_arp23_ca_col.shape[1],
                                                            k_barbed_off_active_arp23_ca_col.shape[1]]))

    def elongate(self, index):
        self.barbed_xyz_mat[index] += self.monomer_length * self.barbed_orientation_mat[index]
        # Enforce periodic boundary conditions
        if np.abs(self.barbed_xyz_mat[index, 0]) > 0.5:
            self.barbed_xyz_mat[index, 0] -= 1.0 * np.sign(self.barbed_xyz_mat[index, 0])
        if np.abs(self.barbed_xyz_mat[index, 1]) > 0.5:
            self.barbed_xyz_mat[index, 1] -= 1.0 * np.sign(self.barbed_xyz_mat[index, 1])

    def cap(self, index):
        self.barbed_is_capped_row[index] = True

    def branch(self, index):
        def rotation_angle_axis(ux_axis, uy_axis, uz_axis, theta_axis):
            r11 = np.cos(theta_axis) + ux_axis ** 2 * (1 - np.cos(theta_axis))
            r12 = ux_axis * uy_axis * (1 - np.cos(theta_axis)) - uz_axis * np.sin(theta_axis)
            r13 = ux_axis * uz_axis * (1 - np.cos(theta_axis)) + uy_axis * np.sin(theta_axis)
            r21 = uy_axis * ux_axis * (1 - np.cos(theta_axis)) + uz_axis * np.sin(theta_axis)
            r22 = np.cos(theta_axis) + uy_axis ** 2 * (1 - np.cos(theta_axis))
            r23 = uy_axis * uz_axis * (1 - np.cos(theta_axis)) - ux_axis * np.sin(theta_axis)
            r31 = uz_axis * ux_axis * (1 - np.cos(theta_axis)) - uy_axis * np.sin(theta_axis)
            r32 = uz_axis * uy_axis * (1 - np.cos(theta_axis)) + ux_axis * np.sin(theta_axis)
            r33 = np.cos(theta_axis) + uz_axis ** 2 * (1 - np.cos(theta_axis))
            rotation_mat = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
            return rotation_mat

        ux_old, uy_old, uz_old = self.barbed_xyz_mat[index]

        # Find an axis perpendicular to orientation of ended end.
        u_perp_mag = np.sqrt(2 + (ux_old + uy_old) ** 2)
        ux_perp_old = 1.0 / u_perp_mag
        uy_perp_old = 1.0 / u_perp_mag
        uz_perp_old = -(ux_old + uy_old) / u_perp_mag

        # Perform rotation to find new orientation.
        theta_polar = self.branch_angle_mu + self.branch_angle_sigma * np.random.randn()
        theta_azi = 2 * np.pi * np.random.rand()
        polar_rotation_mat = rotation_angle_axis(ux_perp_old, uy_perp_old, uz_perp_old, theta_polar)
        u_new_polar_row = polar_rotation_mat @ np.array([ux_old, uy_old, uz_old])
        azi_rotation_mat = rotation_angle_axis(ux_old, uy_old, uz_old, theta_azi)
        u_new_row = azi_rotation_mat @ u_new_polar_row

        # Do it until it's facing the right way (-z).
        while u_new_row[2] >= 0.0:
            theta_polar = self.branch_angle_mu + self.branch_angle_sigma * np.random.randn()
            theta_azi = 2 * np.pi * np.random.rand()
            polar_rotation_mat = rotation_angle_axis(ux_perp_old, uy_perp_old, uz_perp_old, theta_polar)
            u_new_polar_row = polar_rotation_mat @ np.array([ux_old, uy_old, uz_old])
            azi_rotation_mat = rotation_angle_axis(ux_old, uy_old, uz_old, theta_azi)
            u_new_row = azi_rotation_mat @ u_new_polar_row

        # Add new barbed end to relevant arrays.
        self.barbed_xyz_mat = np.vstack((self.barbed_xyz_mat, self.barbed_xyz_mat[index]))
        self.barbed_orientation_mat = np.vstack((self.barbed_orientation_mat, u_new_row))
        self.barbed_is_capped_row = np.append(self.barbed_is_capped_row, False)
        self.barbed_has_wh2_row = np.append(self.barbed_has_wh2_row, False)
        self.barbed_has_monomer_wh2_row = np.append(self.barbed_has_monomer_wh2_row, False)
        self.barbed_has_weak_arp23_ca_row = np.append(self.barbed_has_weak_arp23_ca_row, False)
        self.barbed_has_strong_arp23_ca_row = np.append(self.barbed_has_strong_arp23_ca_row, False)
        self.barbed_has_active_arp23_ca_row = np.append(self.barbed_has_active_arp23_ca_row, False)
        self.barbed2npf_index_row = np.append(self.barbed2npf_index_row, -1)
        self.no_barbed += 1

    def gillespie_step(self):
        self.calculate_transition_rates()
        sum_transition_rate = self.transition_rate_mat.sum()
        random_rate = sum_transition_rate * np.random.rand()
        rate_index = searchsorted_left(self.transition_rate_mat.cumsum(), random_rate)
        rate_row, rate_col = np.unravel_index(rate_index, self.transition_rate_mat.shape)
        transition_index = searchsorted_right(self.transition_rate_edge_row, rate_col)
        if transition_index == 0:
            self.elongate(rate_row)
        elif transition_index == 1:
            self.cap(rate_row)
        elif transition_index == 2:
            self.wh2_has_monomer_row[rate_col - self.transition_rate_edge_row[1]] = True
        elif transition_index == 3:
            self.wh2_has_monomer_row[rate_col - self.transition_rate_edge_row[2]] = False
        elif transition_index == 4:
            self.barbed_has_wh2_row[rate_row] = True
            self.barbed2npf_index_row[rate_row] = rate_col - self.transition_rate_edge_row[3]
            self.wh2_has_barbed_row[rate_col - self.transition_rate_edge_row[3]] = True
        elif transition_index == 5:
            self.barbed_has_wh2_row[rate_row] = False
            self.barbed2npf_index_row[rate_row] = -1
            self.wh2_has_barbed_row[rate_col - self.transition_rate_edge_row[4]] = False
        elif transition_index == 6:
            self.barbed_has_monomer_wh2_row[rate_row] = True
            self.barbed2npf_index_row[rate_row] = rate_col - self.transition_rate_edge_row[5]
            self.wh2_has_monomer_barbed_row[rate_col - self.transition_rate_edge_row[5]] = True
        elif transition_index == 7:
            self.barbed_has_monomer_wh2_row[rate_row] = False
            self.barbed2npf_index_row[rate_row] = -1
            self.wh2_has_monomer_barbed_row[rate_col - self.transition_rate_edge_row[6]] = False
            self.elongate(rate_row)
            self.wh2_has_monomer_row[rate_col - self.transition_rate_edge_row[6]] = False
        elif transition_index == 8:
            self.ca_has_arp23_row[rate_col - self.transition_rate_edge_row[7]] = True
        elif transition_index == 9:
            self.ca_has_arp23_row[rate_col - self.transition_rate_edge_row[8]] = False
        elif transition_index == 10:
            urand = np.random.rand()
            if urand < 0.97:
                self.barbed_has_weak_arp23_ca_row[rate_row] = True
            elif urand < 0.99:
                self.barbed_has_strong_arp23_ca_row[rate_row] = True
            else:
                self.barbed_has_active_arp23_ca_row[rate_row] = True
            self.barbed2npf_index_row[rate_row] = rate_col - self.transition_rate_edge_row[9]
            self.ca_arp23_has_barbed_row[rate_col - self.transition_rate_edge_row[9]] = True
        elif transition_index == 11:
            self.barbed_has_weak_arp23_ca_row[rate_row] = False
            self.barbed2npf_index_row[rate_row] = -1
            self.ca_arp23_has_barbed_row[rate_col - self.transition_rate_edge_row[10]] = False
        elif transition_index == 12:
            self.barbed_has_strong_arp23_ca_row[rate_row] = False
            self.barbed2npf_index_row[rate_row] = -1
            self.ca_arp23_has_barbed_row[rate_col - self.transition_rate_edge_row[11]] = False
        elif transition_index == 13:
            self.barbed_has_active_arp23_ca_row[rate_row] = False
            self.barbed2npf_index_row[rate_row] = -1
            self.ca_arp23_has_barbed_row[rate_col - self.transition_rate_edge_row[12]] = False
            self.ca_has_arp23_row[rate_col - self.transition_rate_edge_row[12]] = False
            self.branch(rate_row)
        time_interval = -1 * np.log(np.random.rand()) / sum_transition_rate
        self.current_time += time_interval

    def simulate(self):
        while self.current_time < self.total_time:
            self.gillespie_step()
            self.barbed_xyz_mat[:, 2] -= self.barbed_xyz_mat[:, 2].min()

    def display(self):
        arrow_length = 0.1 * self.barbed_xyz_mat[:, 2].max()
        fig1_hand = plt.figure()
        axes1_hand = fig1_hand.add_subplot(111, projection='3d')
        axes1_hand.quiver(self.barbed_xyz_mat[:, 0], self.barbed_xyz_mat[:, 1], self.barbed_xyz_mat[:, 2],
                          self.barbed_orientation_mat[:, 0], self.barbed_orientation_mat[:, 1],
                          self.barbed_orientation_mat[:, 2], length=arrow_length)
        return fig1_hand, axes1_hand
