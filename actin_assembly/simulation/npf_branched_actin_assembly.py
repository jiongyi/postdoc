from numpy import pi, linspace, zeros, cos, sin, max, append, nan, sqrt, argmin, min, all, isnan, isin, any
from numpy.random import rand, randn, randint

class network(object):
    def __init__(self, no_filaments = 200, no_npfs = 50, elong_rate = 5.0, branch_rate = 1.0, cap_rate = 0.05, off_rate = 3.0):
        # Copy argument values.
        self.init_no_filaments = no_filaments
        self.no_npfs = no_npfs
        self.elong_rate = elong_rate
        self.branch_rate = branch_rate
        self.cap_rate = cap_rate
        self.off_rate = off_rate

        # Define constants.
        self.LEADING_EDGE_WIDTH = 1000.0
        self.MONOMER_LENGTH = 2.7
        self.MU_THETA = 70 / 180 * pi
        self.SIGMA_THETA = 5 / 180 * pi
        self.TIME_INTERVAL = 1e-2

        # Initialize variables.
        self.no_filaments = self.init_no_filaments
        self.time = 0.0
        self.x_pointed_row = zeros(self.init_no_filaments)
        self.y_pointed_row = self.LEADING_EDGE_WIDTH * rand(self.init_no_filaments)
        self.theta_row = pi * rand(self.init_no_filaments) - 0.5 * pi
        self.u_row = self.MONOMER_LENGTH * cos(self.theta_row)
        self.v_row = self.MONOMER_LENGTH * sin(self.theta_row)
        self.x_barbed_row = self.x_pointed_row + self.u_row
        self.y_barbed_row = self.y_pointed_row + self.v_row
        self.index_proximal_npf_row = zeros(self.init_no_filaments)
        self.is_capped_row = zeros(self.init_no_filaments, dtype = bool)
        self.x_leading_edge = max(self.x_barbed_row)
        self.npf_pos_row = linspace(0.0, self.LEADING_EDGE_WIDTH, self.no_npfs)
        self.npf_state_mat = zeros((self.no_npfs, 3))
        self.npf_state_mat[:, 0] = 1

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
        self.index_proximal_npf_row = append(self.index_proximal_npf_row, nan)
        self.no_filaments += 1
        
    def enforce_boundary_conditions(self, index):
        if self.y_barbed_row[index] > self.LEADING_EDGE_WIDTH:
            self.y_barbed_row[index] -= self.LEADING_EDGE_WIDTH
            self.y_pointed_row[index] = 0.0
            self.x_pointed_row[index] = self.x_barbed_row[index]
        if self.y_barbed_row[index] < 0.0:
            self.y_barbed_row[index] += self.LEADING_EDGE_WIDTH
            self.y_pointed_row[index] = self.LEADING_EDGE_WIDTH
            self.x_pointed_row[index] = self.x_barbed_row[index]
        
        
    def diffuse_barbed_ends(self):
        for i in range(self.no_filaments):
            # Check if tethered.
            if any(isin(i, self.index_proximal_npf_row)):
                if self.npf_state_mat[i, 1] == -1:
                    continue
            else:
                rand_disp = 0.0 * self.MONOMER_LENGTH * randn()
                self.y_barbed_row[i] += rand_disp
                self.y_pointed_row[i]+= rand_disp
                self.enforce_boundary_conditions(i)
            

    def elongate(self, index):
        self.x_barbed_row[index] += self.u_row[index]
        self.y_barbed_row[index] += self.v_row[index]
        self.enforce_boundary_conditions(index)

    def cap(self, index):
        self.is_capped_row[index] = True

    def update_proximity(self):
        for i in range(self.no_filaments):
            i_distance_row = sqrt((self.x_leading_edge - self.x_barbed_row[i])**2 + (self.npf_pos_row - self.y_barbed_row[i])**2)
            i_min_distance = min(i_distance_row)
            if i_min_distance <= (self.MONOMER_LENGTH):
                self.index_proximal_npf_row[i] = argmin(i_distance_row)
            else:
                self.index_proximal_npf_row[i] = nan

    def update(self):
        # Check npfs.
        for i in range(self.no_npfs):
            if self.npf_state_mat[i, 1] == 1:
                if rand() <= (self.off_rate * self.TIME_INTERVAL):
                    self.npf_state_mat[i, 1] = 0
            if self.npf_state_mat[i, 0] == 1:
                if self.npf_state_mat[i, 1] == 0:
                    if rand() <= (self.off_rate * self.TIME_INTERVAL):
                        # Transfer actin to WH2 domain
                        self.npf_state_mat[i, 0] = 0
                        self.npf_state_mat[i, 1] = 1
                        continue
                    elif rand() <= (self.off_rate * self.TIME_INTERVAL):
                        # Lose monomer.
                        self.npf_state_mat[i, 0] = 0
            elif rand() <= (self.elong_rate * self.TIME_INTERVAL):
                # Load profilin-actin complex
                self.npf_state_mat[i, 0] = 1

        for i in range(self.no_npfs):
            if any(isin(i, self.index_proximal_npf_row)):
                continue
            elif self.npf_state_mat[i, 1] == -1:
                self.npf_state_mat[i, 1] = 0
                continue

        for i in range(self.no_npfs):
            if self.npf_state_mat[i, 2] == 0:
                if rand() <= (self.elong_rate * self.TIME_INTERVAL):
                    if rand() <= 0.98:
                        # Load Arp2/3 complex in inactive state.
                        self.npf_state_mat[i, 2] = 1
                        continue
                    else:
                        # Load Arp2/3 complex in active state.
                        self.npf_state_mat[i, 2] = 2
                        continue
            elif self.npf_state_mat[i, 2] == 1:
                if rand() <= (10.0 * self.TIME_INTERVAL):
                    self.npf_state_mat[i, 2] = 0
                    continue
                elif rand() <= (1.0 * self.TIME_INTERVAL):
                    self.npf_state_mat[i, 2] = 2
                    continue
            elif self.npf_state_mat[i, 2] == 2:
                if rand() <= (1.0 * self.TIME_INTERVAL):
                    self.npf_state_mat[i, 2] = 0
                    continue

        # Check barbed ends.
        for i in range(self.no_filaments):
            # Check solution reactions.
            if isnan(self.index_proximal_npf_row[i]):
                if self.is_capped_row[i]:
                    continue
                elif rand() <= (self.elong_rate * self.TIME_INTERVAL):
                    self.elongate(i)
                    continue
                elif rand() <= (self.cap_rate * self.TIME_INTERVAL):
                    self.cap(i)
                    continue
            elif self.is_capped_row[i]:
                continue
            elif rand() <= (self.elong_rate * self.TIME_INTERVAL):
                self.elongate(i)
            elif rand() <= (self.cap_rate * self.TIME_INTERVAL):
                self.cap(i)
                continue
            else:
                i_npf = int(self.index_proximal_npf_row[i])
                if all(self.npf_state_mat[i_npf, :] == [1, -1, 0]):
                    if rand() <= (self.off_rate * self.TIME_INTERVAL):
                        self.npf_state_mat[i_npf, 0] = 0
                        self.elongate(i)
                        continue
                elif all(self.npf_state_mat[i_npf, :] == [1, 0, 0]):
                    rand_int = randint(0, 2)
                    if rand_int == 0:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, :] = [0, 0, 0]
                            self.elongate(i)
                            continue
                    elif rand_int == 1:
                        self.npf_state_mat[i_npf, :] = [1, -1, 0]
                        continue
                elif all(self.npf_state_mat[i_npf, :] == [1, 1, 0]):
                    rand_int = randint(0, 2)
                    if rand_int == 0:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, :] = [0, 1, 0]
                            self.elongate(i)
                            continue
                    elif rand_int == 1:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, :] = [1, 0, 0]
                            self.elongate(i)
                            continue
                elif all(self.npf_state_mat[i_npf, :] == [0, -1, 0]):
                    continue
                elif all(self.npf_state_mat[i_npf, :] == [0, 0, 0]):
                    self.npf_state_mat[i_npf, :] = [0, -1, 0]
                    continue
                elif all(self.npf_state_mat[i_npf, :] == [0, 1, 0]):
                    if rand() <= (self.off_rate * self.TIME_INTERVAL):
                        self.npf_state_mat[i_npf, :] = [0, 0, 0]
                        self.elongate(i)
                        continue
                elif all(self.npf_state_mat[i_npf, :] == [1, -1, 1]):
                    self.npf_state_mat[i_npf, 1] = 0
                    if rand() <= (self.off_rate * self.TIME_INTERVAL):
                        self.npf_state_mat[i_npf, 0] = 0
                        self.elongate(i)
                        continue
                elif all(self.npf_state_mat[i_npf, :] == [1, 0, 1]):
                    rand_int = randint(0, 2)
                    if rand_int == 0:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, :] = [0, 0, 1]
                            self.elongate(i)
                            continue
                    elif rand_int == 1:
                        self.npf_state_mat[i_npf, :] = [1, -1, 1]
                        continue
                elif all(self.npf_state_mat[i_npf, :] == [1, 1, 1]):
                    rand_int = randint(0, 1)
                    if rand_int == 0:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, :] = [0, 1, 1]
                            self.elongate(i)
                            continue
                    elif rand_int == 1:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, :] = [1, 0, 1]
                            self.elongate(i)
                            continue
                elif all(self.npf_state_mat[i_npf, :] == [0, -1, 1]):
                    continue
                elif all(self.npf_state_mat[i_npf, :] == [0, 0, 1]):
                    self.npf_state_mat[i_npf, :] = [0, -1, 1]
                    continue
                elif all(self.npf_state_mat[i_npf, :] == [0, 1, 1]):
                    if rand() <= (self.off_rate * self.TIME_INTERVAL):
                        self.npf_state_mat[i_npf, :] = [0, 0, 1]
                        self.elongate(i)
                        continue
                # Check when Arp2/3 is activated.
                elif all(self.npf_state_mat[i_npf, :] == [1, -1, 2]):
                    self.npf_state_mat[i_npf, 1] = 0
                    if rand() <= (self.off_rate * self.TIME_INTERVAL):
                        self.npf_state_mat[i_npf, 0] = 0
                        self.elongate(i)
                        continue
                elif all(self.npf_state_mat[i_npf, :] == [1, 0, 2]):
                    rand_int = randint(0, 2)
                    if rand_int == 0:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, 0] = 0
                            self.elongate(i)
                            continue
                    elif rand_int == 1:
                        self.npf_state_mat[i_npf, 1] = -1
                        continue
                elif all(self.npf_state_mat[i_npf, :] == [1, 1, 2]):
                    rand_int = randint(0, 1)
                    if rand_int == 0:
                        if rand() <= (self.off_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, 0] = 0
                            self.elongate(i)
                            continue
                    elif rand_int == 1:
                        if rand() <= (self.branch_rate * self.TIME_INTERVAL):
                            self.npf_state_mat[i_npf, :] = [1, 0, 0]
                            self.branch(i)
                            continue
                elif all(self.npf_state_mat[i_npf, :] == [0, -1, 2]):
                    continue
                elif all(self.npf_state_mat[i_npf, :] == [0, 0, 2]):
                    self.npf_state_mat[i_npf, :] = [0, -1, 2]
                    continue
                elif all(self.npf_state_mat[i_npf, :] == [0, 1, 2]):
                    if rand() <= (self.branch_rate * self.TIME_INTERVAL):
                        self.npf_state_mat[i_npf, :] = [0, 0, 0]
                        self.branch(i)
                        continue
        self.time += self.TIME_INTERVAL
        self.x_leading_edge = max(self.x_barbed_row)
        self.diffuse_barbed_ends()
        self.update_proximity()

    def simulate(self, total_time):
        self.update_proximity()
        while self.time <= total_time and sum(~self.is_capped_row) > 0:
            self.update()
        self.maximum_displacement = self.elong_rate * self.MONOMER_LENGTH * self.time
