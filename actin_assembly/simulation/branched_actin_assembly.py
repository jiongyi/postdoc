from numpy import pi, zeros, cos, sin, max, append, array, exp
from numpy.random import rand, randn

class network(object):
    def __init__(self, no_filaments = 200, tether_rate = 1.0, elong_rate = 6.17, branch_rate = 2.0, cap_rate = 0.03):
        # Copy parameter values.
        self.init_no_filaments = no_filaments
        self.tether_rate = tether_rate
        self.elong_rate = elong_rate
        self.branch_rate = branch_rate
        self.cap_rate = cap_rate

        # Define constants.
        self.LEADING_EDGE_WIDTH = 1000.0
        self.MONOMER_WIDTH = 2.7
        self.MU_THETA = 70 / 180 * pi
        self.SIGMA_THETA = 5 / 180 * pi
        self.TIME_INTERVAL = 1e-3
        self.STIFF_COEFF = 1.0
        self.TETHER_FORCE = 10.0 # in pN.

        # Initialize filaments.
        self.no_filaments = self.init_no_filaments
        self.time = 0.0
        self.x_pointed_row = zeros(self.init_no_filaments)
        self.y_pointed_row = self.LEADING_EDGE_WIDTH * rand(self.init_no_filaments)
        self.theta_row = pi * rand(self.init_no_filaments) - 0.5 * pi
        self.u_row = self.MONOMER_WIDTH * cos(self.theta_row)
        self.v_row = self.MONOMER_WIDTH * sin(self.theta_row)
        self.x_barbed_row = self.x_pointed_row + self.u_row
        self.y_barbed_row = self.y_pointed_row + self.v_row
        self.is_tethered_row = zeros(self.init_no_filaments, dtype = bool)
        self.tether_bond_length_row = zeros(self.init_no_filaments)
        self.tether_force_row = zeros(self.init_no_filaments)
        self.tether_force_weight_row = zeros(self.init_no_filaments)
        self.ratchet_force_weight = 1.0
        self.is_capped_row = zeros(self.init_no_filaments, dtype = bool)
        self.x_leading_edge = max(self.x_barbed_row)
        self.is_proximal_row = zeros(self.init_no_filaments, dtype = bool)
        self.no_working = 0

    # Define functions.
    def tether(self, index):
        self.is_tethered_row[index] = True

    def update_tether_force_weight(self):
        self.tether_bond_length_row = self.x_leading_edge - self.x_barbed_row
        self.is_proximal_row[self.tether_bond_length_row <= self.MONOMER_WIDTH] = True
        self.is_proximal_row[self.tether_bond_length_row > self.MONOMER_WIDTH] = False
        self.is_proximal_row[self.is_tethered_row] = True
        self.tether_bond_length_row[~self.is_proximal_row | ~self.is_tethered_row] = 0.0
        self.tether_force_row = self.is_tethered_row * self.STIFF_COEFF * self.tether_bond_length_row
        self.tether_force_weight_row = exp(self.tether_force_row / self.TETHER_FORCE)

    def update_ratchet_force_weight(self):
        self.no_working = sum(self.is_proximal_row * ~self.is_tethered_row * ~self.is_capped_row)
        if self.no_working == 0:
            self.average_ratchet_force = 0
        else:
            self.average_ratchet_force = sum(self.tether_force_row[self.is_tethered_row]) / self.no_working
        self.ratchet_force_weight = exp(-self.average_ratchet_force * self.MONOMER_WIDTH / 4.114)

    def cap(self, index):
        self.is_capped_row[index] = True

    def break_tether(self, index):
        # Break tether.
        self.is_tethered_row[index] = False
        self.tether_bond_length_row[index]  = 0.0
        self.tether_force_row[index] = 0.0
        self.tether_force_weight_row[index] = 0.0

    def branch(self, index):
        theta = self.MU_THETA + self.SIGMA_THETA * randn()
        u_index = self.u_row[index]
        v_index = self.v_row[index]
        u_new = u_index * cos(theta) - v_index * sin(theta)
        # Make sure branch points towards the leading edge.
        if u_new > 0:
            v_new = u_index * sin(theta) + v_index * cos(theta)
        else:
            u_new = u_index * cos(theta) + v_index * sin(theta)
            v_new = -u_index * sin(theta) + v_index * cos(theta)
        # Add new branch to arrays.
        self.x_pointed_row = append(self.x_pointed_row, self.x_barbed_row[index])
        self.y_pointed_row = append(self.y_pointed_row, self.y_barbed_row[index])
        self.x_barbed_row = append(self.x_barbed_row, self.x_barbed_row[index])
        self.y_barbed_row = append(self.y_barbed_row, self.y_barbed_row[index])
        self.u_row = append(self.u_row, u_new)
        self.v_row = append(self.v_row, v_new)
        self.is_capped_row = append(self.is_capped_row, False)
        self.is_tethered_row = append(self.is_tethered_row, False)
        self.is_proximal_row = append(self.is_proximal_row, False)
        self.tether_bond_length_row = append(self.tether_bond_length_row, 0.0)
        self.tether_force_row = append(self.tether_force_row, 0.0)
        self.tether_force_weight_row = append(self.tether_force_weight_row, 0.0)
        self.no_filaments += 1

    def elongate(self, index):
        self.x_barbed_row[index] += self.u_row[index]
        self.y_barbed_row[index] += self.v_row[index]
        # Enforce periodic boundary conditions in the y direction.
        if self.y_barbed_row[index] > self.LEADING_EDGE_WIDTH:
            self.y_barbed_row[index] -= self.LEADING_EDGE_WIDTH
            self.y_pointed_row[index] = 0.0
            self.x_pointed_row[index] = self.x_barbed_row[index]
        if self.y_barbed_row[index] < 0.0:
            self.y_barbed_row[index] += self.LEADING_EDGE_WIDTH
            self.y_pointed_row[index] = self.LEADING_EDGE_WIDTH
            self.x_pointed_row[index] = self.x_barbed_row[index]

    def update(self):
        # Iterate over each barbed end.
        for i in range(self.no_filaments):
            if self.is_proximal_row[i]:
                if self.is_tethered_row[i]:
                    if rand() <= (self.tether_force_weight_row[i] * self.branch_rate * self.TIME_INTERVAL):
                        self.break_tether(i)
                        if rand() <= 0.01:
                            self.branch(i)
                        continue
                elif rand() <= (self.tether_rate * self.TIME_INTERVAL):
                    self.tether(i)
                    continue
                elif self.is_capped_row[i]:
                    continue
                elif rand() <= (self.ratchet_force_weight * self.cap_rate * self.TIME_INTERVAL):
                    self.cap(i)
                    continue
                elif rand() <= (self.ratchet_force_weight * self.elong_rate * self.TIME_INTERVAL):
                    self.elongate(i)
                    continue
            elif self.is_capped_row[i]:
                continue
            elif rand() <= (self.cap_rate * self.TIME_INTERVAL):
                self.cap(i)
                continue
            elif rand() <= (self.elong_rate * self.TIME_INTERVAL):
                self.elongate(i)
                continue

        self.time += self.TIME_INTERVAL
        self.x_leading_edge = max(self.x_barbed_row)
        self.update_tether_force_weight()
        self.update_ratchet_force_weight()

    def simulate(self, total_time):
        self.update_tether_force_weight()
        self.update_ratchet_force_weight()
        while self.time <= total_time and sum(~self.is_capped_row) > 0:
            self.update()
