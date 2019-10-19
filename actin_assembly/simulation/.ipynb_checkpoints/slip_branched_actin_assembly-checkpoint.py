from numpy import pi, zeros, ones, cos, sin, max, append, array, exp
from numpy.random import rand, randn

class network(object):
    def __init__(self, no_filaments = 200, elong_rate = 50, branch_rate = 0.5, cap_rate = 0.5):
        # Copy parameter values.
        self.init_no_filaments = no_filaments
        self.elong_rate = elong_rate
        self.branch_rate = branch_rate
        self.cap_rate = cap_rate

        # Define constants.
        self.LEADING_EDGE_WIDTH = 1000.0
        self.MONOMER_WIDTH = 2.7
        self.MU_THETA = 70 / 180 * pi
        self.SIGMA_THETA = 5 / 180 * pi
        self.TIME_INTERVAL = 1e-3

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
        self.is_capped_row = zeros(self.init_no_filaments, dtype = bool)
        self.x_leading_edge = max(self.x_barbed_row)
        self.is_proximal_row = zeros(self.init_no_filaments, dtype = bool)
        self.elongation_force = 0.0

    # Define functions.
    def update_proximity(self):
        distance_row = self.x_leading_edge - self.x_barbed_row
        self.is_proximal_row[distance_row <= self.MONOMER_WIDTH] = True
        self.is_proximal_row[distance_row > self.MONOMER_WIDTH] = False
        
    def update_slip_weight(self):
        no_proximal = sum(self.is_proximal_row)
        average_force = self.elongation_force / no_proximal
        self.slip_weight = exp(average_force / 10.0)
        print(self.slip_weight)
        self.elongation_force = 0.0

    def cap(self, index):
        self.is_capped_row[index] = True

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
        self.is_proximal_row = append(self.is_proximal_row, True)
        self.no_filaments += 1
        self.no_monomers_added = 0

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
            if self.is_capped_row[i]:
                if self.is_proximal_row[i]:
                    if rand() <= (self.slip_weight * self.branch_rate * self.TIME_INTERVAL):
                        self.branch(i)
                        continue
                else:
                    continue
            elif self.is_proximal_row[i]:
                if rand() <= (self.slip_weight * self.branch_rate * self.TIME_INTERVAL):
                    self.branch(i)
                    continue
                elif rand() <= (self.elong_rate * self.TIME_INTERVAL):
                    self.elongate(i)
                    self.elongation_force += 4.114 / self.MONOMER_WIDTH * self.u_row[i]
                    continue
                elif rand() <= (self.cap_rate * self.TIME_INTERVAL):
                    self.cap(i)
                    continue
            elif rand() <= (self.elong_rate * self.TIME_INTERVAL):
                self.elongate(i)
                continue
            elif rand() <= (self.cap_rate * self.TIME_INTERVAL):
                self.cap(i)
                continue

        self.time += self.TIME_INTERVAL
        self.x_leading_edge = max(self.x_barbed_row)
        self.update_proximity()
        self.update_slip_weight()

    def simulate(self, total_time):
        self.update_proximity()
        self.update_slip_weight()
        while self.time <= total_time and sum(~self.is_capped_row) > 0:
            self.update()
