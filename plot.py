#!/usr/bin/env python3
import csv, sys
import matplotlib.pyplot as plt
import calibrate_gravity

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# Initial position and velocity
pos = [0., 0., 0.]
vel = [0., 0., 0.]

# Points to plot
points = []

# Delta-time for each reading (seconds)
TIME_STEP = 0.01

gravity = None

# Parse csv file and, for each acceleration, update current velocity and position
file_name = sys.argv[1]
with open(file_name) as csvfile:
    r = csv.reader(csvfile)
    for row in r:
        if gravity is None:
            # Treat the first reading as gravity
            gravity = row
            for i in range(3):
                gravity[i] = float(gravity[i])

        for i in range(3):
            # dv = a * dt
            vel[i] += (float(row[i]) - gravity[i]) * TIME_STEP
            # dx = v * dt
            pos[i] += vel[i] * TIME_STEP

        points.append((pos[0], pos[1], pos[2]))

# Set up 3D plot
plt.rcParams['legend.fontsize'] = 10
axes = plt.figure().gca(projection='3d')
# Quit when we close the plot
plt.gcf().canvas.mpl_connect('close_event', quit)

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
