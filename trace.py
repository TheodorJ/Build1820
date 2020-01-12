import numpy as np
import math

smooth_filter = np.array([[0.05, 0.05, 0.05]] * 20)

# numpy doesn't do 2d convolution :(
def convolve2d(x, filter):
    out = []

    for i in range(x.shape[0] - filter.shape[0] + 1):
        out.append(np.sum(x[i:i+filter.shape[0]] * filter, axis=0))

    return np.array(out)

from calibrate_gravity import get_gravity, remove_gravity

def cos(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))


def load_trace(filename):
    return np.loadtxt(open(filename, "rb"), delimiter=",")

def k_cluster(trace, threshold=math.pi/8):
    # for each timestep

    cosines = []
    running_cluster = []
    clusters = []
    for i in range(trace.shape[0]):

        current_vector = trace[i]

        if running_cluster == []:
            running_cluster = [current_vector]

        cluster_mean = np.mean(np.array(running_cluster), axis=0)

        cosines.append(cos(cluster_mean, current_vector))
        if(cos(cluster_mean, current_vector) > threshold):
            # If this acceleration vector is within threshold of the mean vector
            # of the running cluster, then add it to the cluster
            running_cluster.append(current_vector)
        else:
            # Else, create a new cluster
            clusters.append(cluster_mean)
            running_cluster = [current_vector]

    clusters.append(np.mean(np.array(running_cluster), axis=0))

    clusters = [x for x in clusters if np.linalg.norm(x) > 1.0]
    return clusters

def delta_angles(clusters):
    cosines = []
    c_prev = clusters[0]
    for c in clusters[1:]:
        cosines.append(cos(c_prev, c))

    return cosines

gravity = get_gravity(load_trace("data/rest.csv"))

trace = load_trace("data/down_left.csv")
trace = remove_gravity(trace, gravity)
trace = convolve2d(trace, smooth_filter)
print("")
print(trace.shape)
print(len(k_cluster(trace)))
print(k_cluster(trace))
print(delta_angles(k_cluster(trace)))
