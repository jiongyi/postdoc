from numpy import pi, zeros, cos, sin, max, append, array
from numpy.random import rand, randn

class network(object):
    def __init__(current, no_filaments = 200, elong_rate = 100, branch_rate = 1, cap_rate = 0.1):
        # Copy parameter values.
        current.init_no_filaments = no_filaments
        current.elong_rate = elong_rate
        current.branch_rate = branch_rate
        current.cap_rate = cap_rate

        # Define constants.
        current.LEADING_EDGE_WIDTH = 1000.0
        current.MONOMER_WIDTH = 2.7
        current.MU_THETA = 70 / 180 * pi
        current.SIGMA_THETA = 5 / 180 * pi
        current.TIME_INTERVAL = 1e-3

        # Initialize filaments.
        current.no_filaments = current.init_no_filaments
        current.time = 0.0
        current.x_pointed_row = zeros(current.init_no_filaments)
        current.y_pointed_row = current.LEADING_EDGE_WIDTH * rand(current.init_no_filaments)
        current.theta_row = pi * rand(current.init_no_filaments) - 0.5 * pi
        current.u_row = current.MONOMER_WIDTH * cos(current.theta_row)
        current.v_row = current.MONOMER_WIDTH * sin(current.theta_row)
        current.x_barbed_row = current.x_pointed_row + current.u_row
        current.y_barbed_row = current.y_pointed_row + current.v_row
        current.is_capped_row = zeros(current.init_no_filaments, dtype = bool)
        current.x_leading_edge = max(current.x_barbed_row)

    # Define functions.
    def is_active(current, index):
        if current.x_leading_edge - current.x_barbed_row[index] < current.MONOMER_WIDTH:
            return True
        else:
            return False

    def cap(current, index):
        current.is_capped_row[index] = True

    def branch(current, index):
        theta = current.MU_THETA + current.SIGMA_THETA * randn()
        u_index = current.u_row[index]
        v_index = current.v_row[index]
        u_new = u_index * cos(theta) - v_index * sin(theta)
        # Make sure branch points towards the leading edge.
        if u_new > 0:
            v_new = u_index * sin(theta) + v_index * cos(theta)
        else:
            u_new = u_index * cos(theta) + v_index * sin(theta)
            v_new = -u_index * sin(theta) + v_index * cos(theta)
        # Add new branch to arrays.
        current.x_pointed_row = append(current.x_pointed_row, current.x_barbed_row[index])
        current.y_pointed_row = append(current.y_pointed_row, current.y_barbed_row[index])
        current.x_barbed_row = append(current.x_barbed_row, current.x_barbed_row[index])
        current.y_barbed_row = append(current.y_barbed_row, current.y_barbed_row[index])
        current.u_row = append(current.u_row, u_new)
        current.v_row = append(current.v_row, v_new)
        current.is_capped_row = append(current.is_capped_row, False)
        current.no_filaments += 1

    def elongate(current, index):
        current.x_barbed_row[index] += (current.u_row[index] * current.elong_rate * current.TIME_INTERVAL)
        current.y_barbed_row[index] += (current.v_row[index] * current.elong_rate * current.TIME_INTERVAL)
        # Enforce periodic boundary conditions in the y direction.
        if current.y_barbed_row[index] > current.LEADING_EDGE_WIDTH:
            current.y_barbed_row[index] -= current.LEADING_EDGE_WIDTH
            current.y_pointed_row[index] = 0.0
            current.x_pointed_row[index] = current.x_barbed_row[index]
        if current.y_barbed_row[index] < 0.0:
            current.y_barbed_row[index] += current.LEADING_EDGE_WIDTH
            current.y_pointed_row[index] = current.LEADING_EDGE_WIDTH
            current.x_pointed_row[index] = current.x_barbed_row[index]

    def update(current):
        for i in range(current.no_filaments):
            if rand() <= (current.is_active(i) * current.branch_rate * current.TIME_INTERVAL):
                current.branch(i)
            if current.is_capped_row[i]:
                continue
            elif rand() <= (current.cap_rate * current.TIME_INTERVAL):
                current.cap(i)
                continue
            elif rand() <= (current.elong_rate * current.TIME_INTERVAL):
                current.elongate(i)

        current.time += current.TIME_INTERVAL
        current.x_leading_edge = max(current.x_barbed_row)

    def simulate(current, total_time):
        while current.time <= total_time and sum(~current.is_capped_row) > 0:
            current.update()
