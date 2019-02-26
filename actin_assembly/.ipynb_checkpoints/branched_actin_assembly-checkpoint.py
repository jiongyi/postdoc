from numpy import pi, zeros, cos, max, append, argmin
from numpy.random import rand, randn

class network(object):
    def _init_(self, no_filaments = 200, elong_rate = 100, branch_rate = 1, cap_rate = 0.1):
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
        
        # Define functions.
        def is_active(self, index):
            if self.x_leading_edge - self.x_barbed_row[index] < self.MONOMER_WIDTH:
                return True
            else:
                return False
            
        def cap(self, index):
            self.is_capped_row = True
            
        def branch(self, index):
            theta = self.MU_THETA + self.SIGMA_THETA * randn()
            u_index = self.u_row[index]
            v_index = self.v_row[index]
            u_new = u_index * cos(theta) - v * sin(theta)
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
            self.no_filaments += 1
            
            def elongate(self, index):
                self.x_barbed_row[index] += (self.u_row[index] * self.elong_rate * self.TIME_INTERVAL)
                self.y_barbed_row[index] += (self.v_row[index] * self.elong_rate * self.TIME_INTERVAL)
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
                for i in self.no_filaments:
                    if self.is_capped_row[index]:
                        prob_row = array([rand(),
                                          self.is_active[index] * self.branch_rate * self.TIME_INTERVAL])
                        if argmin(prob_row) == 1:
                            self.branch(index)
                    else:
                        prob_row = array([rand(),
                                          self.elong_rate * self.TIME_INTERVAL,
                                          self.is_active[index] * self.branch_rate * self.TIME_INTERVAL,
                                          self.cap_rate * TIME_INTERVAL])
                        min_index = argmin(prob_row)
                        if min_index == 1:
                            self.elongate(index)
                        elif min_index == 2:
                            self.branch(index)
                        elif min_index == 3:
                            self.cap(index)
                self.time += self.TIME_INTERVAL
                self.x_leading_edge = max(self.x_barbed_row)
                
            def simulate(self, total_time):
                while self.time <= total_time and sum(~self.is_capped_row) > 0:
                    self.update()
                                        
                        
                
            
        
        
        
    