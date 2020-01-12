import numpy as np

# Takes a list of 3d inputs and returns a 3d vector mean
def get_gravity(trace):
    return np.mean(trace, axis=0)

def remove_gravity(trace, gravity):
    return trace - gravity
