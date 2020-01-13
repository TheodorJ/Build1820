#!/usr/bin/env python3
import csv, sys
import numpy as np
import math
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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
    cluster_sizes = []
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
            cluster_sizes.append(len(running_cluster))

            running_cluster = [current_vector]

    clusters.append(np.mean(np.array(running_cluster), axis=0))
    cluster_sizes.append(len(running_cluster))

    new_clusters = []
    new_cluster_sizes = []
    for i in range(len(clusters)):
        if np.linalg.norm(clusters[i]) > 1.0:
            new_clusters.append(clusters[i])
            new_cluster_sizes.append(cluster_sizes[i])

    return new_clusters, new_cluster_sizes

def delta_angles(clusters):
    cosines = []
    c_prev = clusters[0]
    for c in clusters[1:]:
        cosines.append(cos(c_prev, c))

    return cosines

if (len(sys.argv) != 2):
    print("Usage: " + sys.argv[0] + " TRACE.csv")
    exit(1)

file_name = sys.argv[1]
gravity = get_gravity(load_trace("data/rest.csv"))

trace = load_trace(file_name)
trace = remove_gravity(trace, gravity)
trace = convolve2d(trace, smooth_filter)
clusters, cluster_sizes = k_cluster(trace)
assert(len(clusters) == len(cluster_sizes))

print("")
print(trace.shape)
print(len(clusters))
print(clusters)
print(delta_angles(clusters))

# Plot trace

# Set up 3D plot
plt.rcParams['legend.fontsize'] = 10
axes = plt.figure().gca(projection='3d')
# Quit when we close the plot
plt.gcf().canvas.mpl_connect('close_event', quit)

# Turn clusters into a series of points
# Initial position and velocity
pos = [0., 0., 0.]
vel = [0., 0., 0.]
# Points to plot
points = []
# Delta-time for each reading (seconds)
TIME_STEP = 0.01

for cluster_idx in range(len(clusters)):
    for i in range(3):
        # dv = a * dt
        vel[i] += (float(clusters[cluster_idx][i]) *
                   TIME_STEP *
                   cluster_sizes[cluster_idx])
        # dx = v * dt
        pos[i] += vel[i] * TIME_STEP * cluster_sizes[cluster_idx]
    points.append((pos[0], pos[1], pos[2]))

# Add parametric curve to plot
x = [(pt[0]) for pt in points]
y = [(pt[1]) for pt in points]
z = [(pt[2]) for pt in points]

print("Plotting " + str(len(points)) + " elements")

axes.plot(x, y, z, label="path", marker=".")
axes.legend()

# Actually display
plt.draw()
plt.pause(2)
input("Press Enter to close")
