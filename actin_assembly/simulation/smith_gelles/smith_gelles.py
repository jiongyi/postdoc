from numpy import linspace, pi, array, cos, sin, newaxis, zeros, full, mod, sign, hstack, vstack, sqrt, all, arange, exp, arctan
from numpy.random import rand, randn
from scipy.spatial.distance import cdist
from matplotlib.pyplot import subplots
from matplotlib.patches import Rectangle

class Network(object):
    def __init__(self, 
                 actin_conc = 5.0,
                 arp23_conc = 50.0e-3,
                 cp_conc = 200.0e-3,
                 npf_density = 1000.0, 
                 total_time=20):
        
        self.current_time = 0.0
        self.time_interval = 1e-3
        self.total_time = total_time
        
        # NPF rates.
        self.WH2_loading_rate = 5.5 * actin_conc * (42.0 / (42.0 + 11.0))**(1/3)
        self.WH2_unloading_rate = 3.0
        self.WH2_untethering_rate = 10.0
        self.CA_loading_rate = 160.0 * arp23_conc
        self.CA_unloading_rate = 10.0
        self.CA_untethering_rate = 1.0
        
        # Solution rates.
        self.elongation_rate = 11 * actin_conc
        self.capping_rate = 2.6 * cp_conc
        
        self.barbed_diff_coeff = 1e-3
        self.monomer_width = 2.7e-3 # in microns
        self.stiff_coeff = 1.0e3 # in pN/um
        self.thermal_tether_force = 10.0
        self.leading_edge_pos = 0.0
        self.branching_region_width = 2 * self.monomer_width
        self.num_ends = 200
        self.num_npfs = 1000

        self.npf_pos_mat = hstack((zeros(self.num_npfs)[:, newaxis], 
                                  linspace(-0.5, 0.5, self.num_npfs)[:, newaxis]))
        # only WH2 and CA domains for now.
        self.WH2_state_row = zeros(self.num_npfs)
        self.CA_state_row = zeros(self.num_npfs)

        self.barbed_pos_mat = rand(self.num_ends, 2)
        self.barbed_pos_mat[:, 0] *= -self.branching_region_width
        self.barbed_pos_mat[:, 1] -= 0.5
        random_barbed_orientation_row = pi * (rand(self.num_ends) - 0.5)
        self.barbed_uv_mat = self.monomer_width * hstack((cos(random_barbed_orientation_row[:, newaxis]), sin(random_barbed_orientation_row[:, newaxis])))
        self.is_capped_row = zeros(self.num_ends, dtype=bool)
        self.tether_index = full(self.num_ends, -1)
        self.tether_force_row = zeros(self.num_ends)
        self.thermal_untethering_rate = 1.0
        self.barbed2npf_index = full(self.num_ends, -1)
        
        self.num_tethering_events = 0
        self.num_untethering_events = 0

    def elongate(self, end_index):
        self.barbed_pos_mat[end_index, :] += self.barbed_uv_mat[end_index, :]
        self.barbed_pos_mat[end_index, 1] = mod(
                                                self.barbed_pos_mat[end_index, 1],
                                                -0.5 * sign(self.barbed_uv_mat[end_index, 1])
                                                )
    def tether(self, end_index, npf_index, domain_index):
        self.tether_index[end_index] = domain_index
        self.barbed2npf_index[end_index] = npf_index
        self.tether_force_row[end_index] = self.stiff_coeff * (self.leading_edge_pos - self.barbed_pos_mat[end_index, 0])
        if domain_index == 0:
            self.WH2_state_row[npf_index] = -1
        elif domain_index == 1:
            self.CA_state_row[npf_index] = -1
            self.num_tethering_events += 1

    def untether(self, end_index):
        this_npf_index = self.barbed2npf_index[end_index]
        if self.tether_index[end_index] == 0:
            self.WH2_state_row[this_npf_index] = 0 # Leave WH2 empty.
        elif self.tether_index[end_index] == 1:
            self.CA_state_row[this_npf_index] = 0 # Take Arp2/3.
            self.num_untethering_events += 1
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
        
    def metropolis_step(self):
        #Update mechanics
        self.tether_force_row[self.tether_index != -1] = self.stiff_coeff * (self.leading_edge_pos - self.barbed_pos_mat[self.tether_index != -1, 0])
        self.num_loaded_ends = (self.leading_edge_pos - self.barbed_pos_mat[self.tether_index == -1, 0] <= self.monomer_width).sum()
        if self.num_loaded_ends == 0:
            self.mean_load = 0.0
        else:
            self.mean_load = self.tether_force_row.sum() / self.num_loaded_ends
        self.loaded_capping_rate = self.capping_rate * exp(-self.mean_load * self.monomer_width / 4.114)
        self.loaded_elongation_rate = self.elongation_rate * exp(-self.mean_load * self.monomer_width / 4.114)
        WH2_index = arange(self.num_npfs)
        CA_index = arange(self.num_npfs)
        for i in range(self.num_ends):
            if self.tether_index[i] == -1:
                i_npf_dist_row = cdist(self.barbed_pos_mat[i, newaxis] + sqrt(2 * self.barbed_diff_coeff * self.time_interval) * randn(2), self.npf_pos_mat)
                if i_npf_dist_row.min() <= self.monomer_width:
                    i_nearest_npf = i_npf_dist_row.argmin()
                    if self.is_capped_row[i] == False and self.WH2_state_row[i_nearest_npf] == 0 and self.CA_state_row[i_nearest_npf] == 1:
                        if rand() > 0.9:
                            self.tether(i, i_nearest_npf, 0)
                            WH2_index = WH2_index[WH2_index != i_nearest_npf]
                        else:
                            self.tether(i, i_nearest_npf, 1)
                            CA_index = CA_index[CA_index != i_nearest_npf]
                    elif self.is_capped_row[i] == False and self.WH2_state_row[i_nearest_npf] == 0:
                        self.tether(i, i_nearest_npf, 0)
                        WH2_index = WH2_index[WH2_index != i_nearest_npf]
                    elif self.CA_state_row[i_nearest_npf] == 1:
                        self.tether(i, i_nearest_npf, 1)
                        CA_index = CA_index[CA_index != i_nearest_npf]
                elif self.is_capped_row[i] == False:
                    if self.leading_edge_pos - self.barbed_pos_mat[i, 0] <= self.monomer_width:
                        if exp(-self.loaded_capping_rate * self.time_interval) < rand():
                            self.cap(i)
                            continue
                        elif exp(-self.loaded_elongation_rate * self.time_interval) < rand():
                            self.elongate(i)
                    elif exp(-self.capping_rate * self.time_interval) < rand():
                        self.cap(i)
                        continue
                    elif exp(-self.elongation_rate * self.time_interval) < rand():
                        self.elongate(i)

            else:
                i_tethered_npf = self.barbed2npf_index[i]
                if self.tether_index[i] == 0:
                    i_WH2_untethering_rate = self.WH2_untethering_rate * exp(self.tether_force_row[i] / self.thermal_tether_force)
                    if exp(-i_WH2_untethering_rate * self.time_interval) < rand():
                        self.untether(i)
                        WH2_index = WH2_index[WH2_index != i_tethered_npf]
                elif self.tether_index[i] == 1:
                    if 2e-2 < rand():
                        i_CA_untethering_rate = 10 * self.CA_untethering_rate * exp(self.tether_force_row[i] / self.thermal_tether_force)
                        if exp(-i_CA_untethering_rate * self.time_interval) < rand():
                            self.untether(i)
                            CA_index = CA_index[CA_index != i_tethered_npf]
                    else:
                        i_CA_untethering_rate = self.CA_untethering_rate * exp(self.tether_force_row[i] / self.thermal_tether_force)
                        if exp(-i_CA_untethering_rate * self.time_interval) < rand():
                                self.untether(i)
                                CA_index = CA_index[CA_index != i_tethered_npf]
                                if self.WH2_state_row[i_tethered_npf] == 1:
                                    self.branch(i)
                                    self.WH2_state_row[i_tethered_npf] = 0
                                
                        
        for j in WH2_index:
            if self.WH2_state_row[j] == 0:
                if exp(-self.WH2_loading_rate * self.time_interval) < rand():
                    self.WH2_state_row[j] = 1
            elif self.WH2_state_row[j] == 1:
                if exp(-self.WH2_unloading_rate * self.time_interval) < rand():
                    self.WH2_state_row[j] = 0            
        for k in CA_index:
            if self.CA_state_row[k] == 0:
                if exp(-self.CA_loading_rate * self.time_interval) < rand():
                    self.CA_state_row[k] = 1
            elif self.CA_state_row[k] == 1:
                if exp(-self.CA_unloading_rate * self.time_interval) < rand():
                    self.CA_state_row[k] = 0
        
        self.leading_edge_pos = max((self.leading_edge_pos, self.barbed_pos_mat[:, 0].max()))
        self.npf_pos_mat[:, 0] = max((self.leading_edge_pos, self.barbed_pos_mat[:, 0].max()))
        self.current_time += self.time_interval
        
    def simulate(self):
        while self.current_time <= self.total_time and (~self.is_capped_row).sum() > 0:
            self.metropolis_step()
            
    def display(self):
        fig_hand, axes_hand = subplots()
        axes_hand.add_patch(Rectangle((self.leading_edge_pos - self.monomer_width, -0.5), self.monomer_width, 1, facecolor='lightgoldenrodyellow', alpha=0.8))
        axes_hand.quiver(self.barbed_pos_mat[:, 0], self.barbed_pos_mat[:, 1], self.barbed_uv_mat[:, 0], self.barbed_uv_mat[:, 1], color='b')
        axes_hand.tick_params(labelsize=14)
        axes_hand.set_xlabel('x ($\mu$m)', fontsize=14)
        return fig_hand, axes_hand
        
    def plot_orientation(self):
        fig_hand, axes_hand = subplots()
        axes_hand.hist(180 * arctan(self.barbed_uv_mat[200:, 1] / self.barbed_uv_mat[200:, 0]) / pi, [-90, -87.5, -52.5, -17.5, 17.5, 52.5, 87.5, 90])
        return fig_hand, axes_hand