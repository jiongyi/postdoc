from numpy import pi, zeros, cos, sin, copy, amax, flatnonzero, append, exp, sum, arctan, array, histogram, nan, abs
from numpy.random import rand, randn
from matplotlib.pyplot import subplots
from matplotlib.animation import FFMpegWriter
import matplotlib
import datetime

class network(object):
    def __init__(self, nucleation_prob = 3e-2,
                 no_monomer_bind_sites = 2,
                 cap_rate_const = 0.1,
                 simulation_time = 60.0):
        # Copy argument values.
        self.nucleation_prob = nucleation_prob
        self.no_monomer_bind_sites = no_monomer_bind_sites
        self.cap_rate_const = cap_rate_const
        self.simulation_time = simulation_time
        # Define constants.
        self.leading_edge_width = 1e3
        self.monomer_width = 2.7
        self.mu_theta = 70.0 / 180.0 * pi
        self.mu_sigma = 5.0 / 180.0 * pi
        self.sol_elongation_rate_const = 10.0 # subunits per second.
        self.net_elongation_rate_const = (1 + self.no_monomer_bind_sites) * self.sol_elongation_rate_const
        self.nucleation_rate_const = self.nucleation_prob * self.net_elongation_rate_const
        self.untether_rate_const = 1.0
        self.thermal_force = 10.0
        self.spring_constant = 1.0
        # Initialize simulation.
        self.no_filaments = 200
        self.no_monomers = 0
        self.time = 0.0
        self.time_interval = 1e-3
        self.x_point_row = zeros(self.no_filaments)
        self.y_point_row = self.leading_edge_width * rand(self.no_filaments)
        self.theta_row = pi * rand(self.no_filaments) - 0.5 * pi
        self.u_row = cos(self.theta_row)
        self.v_row = sin(self.theta_row)
        self.x_barb_row = copy(self.x_point_row) + self.u_row
        self.y_barb_row = copy(self.y_point_row) + self.v_row
        self.is_capped_row = zeros(self.no_filaments, dtype = bool)
        self.is_tethered_row = zeros(self.no_filaments, dtype = bool)
        self.mean_load = 0.0
        self.loaded_net_elongation_rate_const = copy(self.net_elongation_rate_const)
        self.loaded_cap_rate_const = copy(self.cap_rate_const)
        self.x_leading = amax(self.x_barb_row)
        
    def find_active_barb(self):
        distance_row = self.x_leading - self.x_barb_row
        self.index_active_row = flatnonzero(distance_row <= self.monomer_width)

    def update_mechanics(self):
        distance_row = self.x_leading - self.x_barb_row
        spring_force_row = self.spring_constant * distance_row
        spring_force_row[~self.is_tethered_row] = 0.0
        self.loaded_untether_rate_const_row = self.untether_rate_const * exp(spring_force_row / self.thermal_force)
        no_active_barb = len(self.index_active_row)
        if no_active_barb > 0:
            self.mean_load = sum(spring_force_row) / no_active_barb
            load_weight = exp(-self.mean_load * self.monomer_width / 4.114)
            self.loaded_net_elongation_rate_const = load_weight * self.net_elongation_rate_const
            self.loaded_cap_rate_const = load_weight * self.cap_rate_const
        else:
            self.mean_load = 0.0
            self.loaded_net_elongation_rate_const = copy(self.net_elongation_rate_const)
        # Incorporate angle dependence.
        self.alpha_row = arctan(self.v_row / self.u_row) / pi * 180
        self.alpha_elongation_rate_const_row = self.loaded_net_elongation_rate_const + (self.net_elongation_rate_const - self.loaded_net_elongation_rate_const) / 90.0 * self.alpha_row
        self.alpha_cap_rate_const_row = self.loaded_cap_rate_const + (self.cap_rate_const - self.loaded_cap_rate_const) / 90.0 * self.alpha_row
        self.alpha_nucleation_rate_const_row = self.nucleation_rate_const / cos(self.alpha_row / 180.0 * pi)
    def cap(self, index):
        self.is_capped_row[index] = True
                
    def nucleate(self, index):
        theta_new = self.mu_theta + self.mu_sigma * randn()
        if rand() < 0.5:
            theta_new *= -1.0
        u = self.u_row[index]
        v = self.v_row[index]
        u_new = u * cos(theta_new) - v * sin(theta_new)
        # Make sure branch points towards the leading edge.
        if u_new > 0:
            v_new = u * sin(theta_new) + v * cos(theta_new)
        else:
            u_new = u * cos(theta_new) + v * sin(theta_new)
            v_new = -u * sin(theta_new) + v * cos(theta_new)
        # Add new end to arrays.
        self.x_point_row = append(self.x_point_row, self.x_barb_row[index])
        self.y_point_row = append(self.y_point_row, self.y_barb_row[index])
        self.x_barb_row = append(self.x_barb_row, self.x_barb_row[index])
        self.y_barb_row = append(self.y_barb_row, self.y_barb_row[index])
        self.u_row = append(self.u_row, u_new)
        self.v_row = append(self.v_row, v_new)
        self.is_capped_row = append(self.is_capped_row, False)
        self.is_tethered_row = append(self.is_tethered_row, True)
        self.no_filaments += 1
        self.no_monomers += 1
                
    def elongate(self, index):
        self.x_barb_row[index] += (self.monomer_width * self.u_row[index])
        self.y_barb_row[index] += (self.monomer_width * self.v_row[index])
        # Enforce periodic boundary conditions in the y direction:
        if self.y_barb_row[index] > self.leading_edge_width:
            self.y_barb_row[index] = self.y_barb_row[index] - self.leading_edge_width
            self.y_point_row[index] = 0.0
            self.x_point_row[index] = self.x_barb_row[index]
        if self.y_barb_row[index] < 0.0:
            self.y_barb_row[index] = self.y_barb_row[index] + self.leading_edge_width
            self.y_point_row[index] = self.leading_edge_width
            self.x_point_row[index] = self.x_barb_row[index]
        self.no_monomers += 1
                        
    def update(self):
        self.find_active_barb()
        self.update_mechanics()
        for i_filament in range(self.no_filaments):
            if self.is_tethered_row[i_filament]:
                # Check if it will untether.
                if rand() <= (self.loaded_untether_rate_const_row[i_filament] * self.time_interval):
                    self.is_tethered_row[i_filament] = False
                    continue
            elif i_filament in self.index_active_row:
                # Barbed end is at the membrane.
                if self.is_capped_row[i_filament]:
                    #if rand() <= (self.nucleation_rate_const * self.time_interval):
                    if rand() <= (self.alpha_nucleation_rate_const_row[i_filament] * self.time_interval):
                        self.nucleate(i_filament)
                    continue
                #if rand() <= (self.loaded_cap_rate_const * self.time_interval):
                if rand() <= (self.alpha_cap_rate_const_row[i_filament] * self.time_interval):
                    self.cap(i_filament)
                    continue
                #elif rand() <= (self.loaded_net_elongation_rate_const * self.time_interval):
                elif rand() <= (self.alpha_elongation_rate_const_row[i_filament] * self.time_interval):
                    self.elongate(i_filament)
                #elif rand() <= (self.nucleation_rate_const * self.time_interval):
                elif rand() <= (self.alpha_nucleation_rate_const_row[i_filament] * self.time_interval):
                    self.nucleate(i_filament)
            elif self.is_capped_row[i_filament]:
                continue
            elif rand() <= (self.cap_rate_const * self.time_interval):
                self.cap(i_filament)
                continue
            elif rand() <= (self.sol_elongation_rate_const * self.time_interval):
                self.elongate(i_filament)
                
        self.time += self.time_interval
        self.x_leading = amax(self.x_barb_row)
                    
    def simulate(self):
        while self.time <= self.simulation_time and sum(~self.is_capped_row) > 0:
            self.update()
            
    def plot_network(self):
        network_fig_hand, network_axes_hand = subplots()
        for i_filament in range(self.no_filaments):
            if self.is_capped_row[i_filament]:
                network_axes_hand.plot([self.x_point_row[i_filament], self.x_barb_row[i_filament]],
                                       [self.y_point_row[i_filament], self.y_barb_row[i_filament]],
                                       color = 'red', linewidth = 2)
            else:
                network_axes_hand.plot([self.x_point_row[i_filament], self.x_barb_row[i_filament]],
                                       [self.y_point_row[i_filament], self.y_barb_row[i_filament]],
                                       color = 'blue', linewidth = 2)
        network_axes_hand.set_xlabel("x (nm)", fontsize = 12)
        network_axes_hand.set_ylabel("L (nm)", fontsize = 12)
        return network_fig_hand, network_axes_hand
        
    def compute_alpha_order(self):
        new_alpha_row = self.alpha_row[200:]
        counts_0 = 1.0 * sum(abs(new_alpha_row) <= 17.5)
        counts_35 = 0.5 * sum(abs(new_alpha_row - 35.0) <= 17.5) + 0.5 * sum(abs(new_alpha_row + 35.0) <= 17.5)
        if counts_0 + counts_35 == 0:
            self.alpha_order_param = nan
        else:
            self.alpha_order_param = (counts_0 - counts_35)/ (counts_0 + counts_35)
            
    def plot_alpha_distribution(self):
        alpha_fig_hand, alpha_axes_hand = subplots()
        new_alpha_row = self.alpha_row[200:]
        alpha_axes_hand.hist(new_alpha_row, bins = array([-87.5, -52.5, -17.5, 17.5, 52.5, 87.5]))
        alpha_axes_hand.set_xlabel("Filament orientation", fontsize = 12)
        alpha_axes_hand.set_ylabel("Counts", fontsize = 12)
        return alpha_fig_hand, alpha_axes_hand
    
    def make_movie(self):
        matplotlib.use("Agg")
        # Set up figure.
        movie_fig_hand, movie_axes_hand = subplots()
        movie_axes_hand.plot([self.x_point_row, self.x_barb_row],
                            [self.y_point_row, self.y_barb_row],
                            color = 'blue', linewidth = 2)
        movie_axes_hand.set_xlim(0.0, self.simulation_time * self.net_elongation_rate_const * self.monomer_width)
        movie_axes_hand.set_ylim(0.0, 1000.0)
        movie_axes_hand.set_xlabel("x (nm)", fontsize = 12)
        movie_axes_hand.set_ylabel("y (nm)", fontsize = 12)
        
        # Animate.
        frame_rate = 1
        writer = FFMpegWriter(fps = frame_rate)
        with writer.saving(movie_fig_hand, "network_" + datetime.datetime.now().isoformat() + ".mp4", dpi = 100):
            while self.time <= self.simulation_time:
                next_time = self.time + 1.0 / frame_rate
                while self.time < next_time:
                    self.update()
                del movie_axes_hand.lines[:]
                for i_filament in range(self.no_filaments):
                    if self.is_capped_row[i_filament]:
                        filament_color = 'red'
                    else:
                        filament_color = 'blue'
                    movie_axes_hand.plot([self.x_point_row[i_filament], self.x_barb_row[i_filament]],
                                        [self.y_point_row[i_filament], self.y_barb_row[i_filament]],
                                        color = filament_color, linewidth = 2)
                writer.grab_frame()