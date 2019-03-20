from numpy import std

def z_project(z_stack_mat):
    projected_mat = std(z_stack_mat, axis = -1)
    return projected_mat