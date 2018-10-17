# Import libraries
from numpy import array, copy, flatnonzero, zeros, nan, isnan, pi, cos, sin, mod, int, argmin, append, amax, exp, arctan, abs, logical_and, linspace, sqrt
from numpy.random import rand, poisson, randn, choice

class network(object):
    def __init__(n, npf_density = 50.0, elongation_rate = 6.0, arp23_activation_rate = 6.0e-2, capping_rate = 0.08, total_time = 20.0):
        # Copy parameters
        n.npf_density = npf_density * 1e-3
        n.elongation_rate = elongation_rate
        n.arp23_activation_rate = arp23_activation_rate
        n.capping_rate = capping_rate
        n.total_time = total_time
        # Define constants.
        n.MONOMER_WIDTH = 2.7
        n.EDGE_WIDTH = 1000.0
        n.THETA_MU = 70.0 / 180 * pi
        n.THETA_SIGMA = 5.0 / 180 * pi
        n.TIME_INTERVAL = 1e-3
        
        # Initialize variables.
        n.no_filaments = 200
        n.time = 0.0
        n.x_pointed_row = zeros(n.no_filaments)
        n.y_pointed_row = n.EDGE_WIDTH * rand(n.no_filaments)
        n.theta_row = pi * rand(n.no_filaments) - 0.5 * pi
        n.u_row = n.MONOMER_WIDTH * cos(n.theta_row)
        n.v_row = n.MONOMER_WIDTH * sin(n.theta_row)
        n.x_barbed_row = copy(n.x_pointed_row) + n.u_row
        n.y_barbed_row = copy(n.y_pointed_row) + n.v_row
        n.is_capped_row = zeros(n.no_filaments, dtype = bool)
        n.edge_position = amax(n.x_barbed_row)
    
    def elongate(n, index):
        n.x_barbed_row[index] += n.u_row[index]
        n.y_barbed_row[index] += n.v_row[index]
        # Enforce periodic boundary conditions in the y direction.
        if n.y_barbed_row[index] > n.EDGE_WIDTH:
            n.y_barbed_row[index] -= n.EDGE_WIDTH
            n.y_pointed_row[index] = 0.0
            n.x_pointed_row[index] = n.y_barbed_row[index]
        if n.y_barbed_row[index] < 0.0:
            n.y_barbed_row[index] += n.EDGE_WIDTH
            n.y_pointed_row[index] = n.EDGE_WIDTH
            n.x_pointed_row[index] = n.x_barbed_row[index]
            
    def branch(n, index):
        theta = choice([-1, 1]) * n.THETA_MU + n.THETA_SIGMA * randn()
        u = n.u_row[index]
        v = n.v_row[index]
        u_new = u * cos(theta) - v * sin(theta)
        if u_new > 0:
            v_new = u * sin(theta) + v * cos(theta)
            n.theta_row = append(n.theta_row, theta)
        else:
            u_new = u * cos(theta) + v * sin(theta)
            v_new = -u * sin(theta) + v * cos(theta)
            n.theta_row = append(n.theta_row, -theta)
        # Create pointed end.
        n.x_pointed_row = append(n.x_pointed_row, n.x_barbed_row[index])
        n.y_pointed_row = append(n.y_pointed_row, n.y_barbed_row[index])
        n.x_barbed_row = append(n.x_barbed_row, n.x_barbed_row[index])
        n.y_barbed_row = append(n.y_barbed_row, n.y_barbed_row[index])
        n.u_row = append(n.u_row, u_new)
        n.v_row = append(n.v_row, v_new)
        n.is_capped_row = append(n.is_capped_row, False)
        
    def cap(n, index):
        n.is_capped_row[index] = True
        
    def update(n):
        for i in range(n.no_filaments):
            # Calculate reaction probabilities.
            if n.is_capped_row[i] == True or ((n.edge_position - n.x_barbed_row[i]) <= n.MONOMER_WIDTH):
                elongation_probability = 0.0
            else:
                elongation_probability = n.elongation_rate * n.TIME_INTERVAL
            
            if (n.edge_position - n.x_barbed_row[i]) <= n.MONOMER_WIDTH:
                branching_probability = n.arp23_activation_rate * n.TIME_INTERVAL
            else:
                branching_probability = 0.0
            
            if n.is_capped_row[i] == True or ((n.edge_position - n.x_barbed_row[i]) <= n.MONOMER_WIDTH):
                capping_probability = 0.0
            else:
                capping_probability = n.capping_rate * n.TIME_INTERVAL
            
            if rand() < elongation_probability:
                n.elongate(i)
            if rand() < branching_probability:
                n.branch(i)
            if rand() < capping_probability:
                n.cap(i)
        
        # Update.
        n.time += n.TIME_INTERVAL
        n.edge_position = amax(n.x_barbed_row)
        n.no_filaments = len(n.x_pointed_row)
        
    def simulate(n):
        while n.time <= n.total_time and (sum(~n.is_capped_row) > 0):
            n.update()
        