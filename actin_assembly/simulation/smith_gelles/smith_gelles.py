from numpy import linspace, pi, cos, sin, newaxis, zeros, fill, mod, sign, randn, hstack, vstack, all
from numpy.random import rand
from scipy.spatial.distance import cdist

class Network(object):
    def __init__(self):
        self.monomer_width = 2.7e-3 # in microns
        self.stiff_coeff = 1.0 # in pN/nm.
        self.leading_edge_pos = 0.0
        self.branching_region_width = 2 * self.monomer_width
        self.num_ends = 200
        self.num_npfs = 1000

        self.npf_pos_mat = array([
                                  zeros(self.num_npfs)[:, newaxis],
                                  linspace(-0.5, 0.5, self.num_npfs)[:, newaxis]
                                  ])
        # only WH2 and CA domains for now.
        self.npf_state_mat = zeros((self.num_npfs, 2))

        self.barbed_pos_mat = rand(self.num_ends, 2)
        self.barbed_pos_mat[:, 0] *= -self.branching_region_width
        self.barbed_pos_mat[:, 1] -= 0.5
        random_barbed_orientation_row = pi * (rand(self.num_ends) - 0.5)
        self.barbed_uv_mat = self.monomer_width *
                             array([
                                    cos(random_barbed_orientation_row[:, newaxis]),
                                    sin(random_barbed_orientation_row[:, newaxis])
                                   ])
        self.is_capped_row = zeros(self.num_ends, dtype=bool)
        self.tether_index = fill(self.num_ends, -1)
        self.tether_force_row = zeros(self.num_ends)
        self.barbed2npf_index = fill(self.num_ends, -1)

    def elongate(self, end_index):
        self.barbed_pos_mat[end_index, :] = mod(
                                                self.barbed_pos_mat[end_index, :] +
                                                self.barbed_uv_mat[end_index, :],
                                                -0.5 * sign(self.barbed_uv_mat[end_index, :])
                                                )
    def tether(self, end_index, npf_index, domain_index):
        self.tether_index[end_index] = domain_index
        self.barbed2npf_index[end_index] = npf_index
        self.tether_force_row[end_index] = self.stiff_coeff *
                                           (self.leading_edge_pos -
                                           self.end_pos_mat[end_index, 0])
        self.npf_state_mat[npf_index, domain_index] = -1

    def untether(self, end_index):
        this_npf_index = self.barbed2npf_index[end_index]
        if self.tether_index[end_index] == 0:
            self.npf_state_mat[this_npf_index, 0] = 0 # Leave WH2 empty.
        elif self.tether_index[end_index] == 1:
            self.npf_state_mat[this_npf_index, 1] = 1 # Leave Arp2/3 on CA.
        self.tether_index[end_index] = -1
        self.barbed2npf_index[end_index] = -1
        self.tether_force_row[end_index] = 0.0

    def branch(self, end_index):
        random_theta = pi * (70 + 5 * randn()) / 180
        if rand() >= 0.5:
            random_theta *= -1
        u = self.barbed_uv_mat[end_index, 0]
        v = self.barbed_uv_mat[end_index, 1]
        u_new = u * cos(random_theta) - v * sin(random_theta)
        if u_new > 0:
            v_new = u * sin(random_theta) + v * cos(random_theta)
        else:
            u_new = u * cos(random_theta) + v * sin(random_theta)
            v_new = -u * sin(random_theta) + v * cos(random_theta)

        self.num_ends += 1
        self.barbed_pos_mat = vstack((self.barbed_pos_mat,
                                      self.barbed_pos_mat[end_index]))
        self.barbed_uv_mat = vstack((self.barbed_uv_mat,
                                     array([u_new, v_new])))
        self.is_capped_row = hstack((self.is_capped_row, False))
        self.tether_index = hstack((self.tether_index, -1))
        self.tether_force_row = hstack((self.tether_force_row, 0.0))
        self.barbed2npf_index = hstack((self.barbed2npf_index, -1))

    def cap(self, end_index):
        self.is_capped_row[end_index] = True

    def metropolis_step():
        for i in range(self.num_ends):
            i_end2npf_dist_row = cdist(self.barbed_pos_mat[i, :], self.npf_pos_mat)
            i_min_dist = i_end2npf_dist_row.min()
            if i_min_dist <= self.monomer_width:
                i_nearest_npf_index = i_end2npf_dist_row.argmin()
                if all(self.npf_state_mat[i_nearest_npf_index, :] = [0, 1]):
                    
        for j in range(self.num_npfs):
            j_npf_state_row = self.npf_state_mat[j, :]
            if j_npf_state_row[0] == 0:
                if exp(-self.wh2_loading_rate * self.time_interval) < rand():
                    self.npf_state_mat[j, 0] = 1
            elif j_npf_state_row[0] == 1:
                if exp(-self.wh2_unloading_rate * self.time_interval) < rand():
                    self.npf_state_mat[j, 0] = 0
            if j_npf_state_row[1] == 0:
                if exp(-self.ca_loading_rate * self.time_interval) < rand():
                    self.npf_state_mat[j, 1] = 1
            elif j_npf_state_row[1] == 1:
                if exp(-self.ca_unloading_rate * self.time_interval) < rand():
                    self.npf_state_mat[j, 1] = 0

    def simulate():
