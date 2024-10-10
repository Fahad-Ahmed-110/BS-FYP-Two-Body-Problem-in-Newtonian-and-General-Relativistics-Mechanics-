# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:46:28 2024

@author: Fahad
"""

# Two Body Problem by RK4 Adeptive:


import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)

# Initial conditions
m1 = m2 = 1.0  # masses of both bodies (kg)
t0 = 0.0  # initial time
tf = 10000.0  # final time
h0 = 1.0  # initial time step size
tolerance = 1e-6  # tolerance for adaptive step size control

# Initial positions and velocities
r1_0 = np.array([1.0, 0.0])  # initial position of body 1 (m)
v1_0 = np.array([0.0, 1.0])  # initial velocity of body 1 (m/s)
r2_0 = np.array([-1.0, 0.0])  # initial position of body 2 (m)
v2_0 = np.array([0.0, -1.0])  # initial velocity of body 2 (m/s)

# Function to compute acceleration (force) on each body
def compute_acceleration(r1, r2):
    r = r2 - r1
    r_norm = np.linalg.norm(r)
    a1 = G * m2 / r_norm**3 * r
    a2 = -G * m1 / r_norm**3 * r
    return a1, a2

# RK4 integration step
def rk4_step(r1, v1, r2, v2, h):
    k1v1 = h * compute_acceleration(r1, r2)[0]
    k1v2 = h * compute_acceleration(r2, r1)[1]
    k1r1 = h * v1
    k1r2 = h * v2

    k2v1 = h * compute_acceleration(r1 + 0.5 * k1r1, r2 + 0.5 * k1r2)[0]
    k2v2 = h * compute_acceleration(r2 + 0.5 * k1r2, r1 + 0.5 * k1r1)[1]
    k2r1 = h * (v1 + 0.5 * k1v1)
    k2r2 = h * (v2 + 0.5 * k1v2)

    k3v1 = h * compute_acceleration(r1 + 0.5 * k2r1, r2 + 0.5 * k2r2)[0]
    k3v2 = h * compute_acceleration(r2 + 0.5 * k2r2, r1 + 0.5 * k2r1)[1]
    k3r1 = h * (v1 + 0.5 * k2v1)
    k3r2 = h * (v2 + 0.5 * k2v2)

    k4v1 = h * compute_acceleration(r1 + k3r1, r2 + k3r2)[0]
    k4v2 = h * compute_acceleration(r2 + k3r2, r1 + k3r1)[1]
    k4r1 = h * (v1 + k3v1)
    k4r2 = h * (v2 + k3v2)

    r1_new = r1 + (k1r1 + 2 * k2r1 + 2 * k3r1 + k4r1) / 6
    v1_new = v1 + (k1v1 + 2 * k2v1 + 2 * k3v1 + k4v1) / 6
    r2_new = r2 + (k1r2 + 2 * k2r2 + 2 * k3r2 + k4r2) / 6
    v2_new = v2 + (k1v2 + 2 * k2v2 + 2 * k3v2 + k4v2) / 6

    return r1_new, v1_new, r2_new, v2_new

# Adaptive RK4 integration
def adaptive_rk4(r1, v1, r2, v2, t, h):
    while t < tf:
        # Perform one step of RK4 with step size h
        r1_1, v1_1, r2_1, v2_1 = rk4_step(r1, v1, r2, v2, h)
        # Perform two steps of RK4 with step size h/2
        r1_2, v1_2, r2_2, v2_2 = rk4_step(r1, v1, r2, v2, h / 2)
        r1_2, v1_2, r2_2, v2_2 = rk4_step(r1_2, v1_2, r2_2, v2_2, h / 2)

        # Compute the error estimate
        error = np.linalg.norm(r1_2 - r1_1) + np.linalg.norm(r2_2 - r2_1)

        # If the error is within the tolerance, accept the step
        if error < tolerance:
            r1, v1, r2, v2 = r1_2, v1_2, r2_2, v2_2
            t += h
            yield t, r1, r2
        # Adjust the step size
        h *= min(5.0, max(0.2, 0.8 * (tolerance / error)**0.2))

# Arrays to store positions over time
r1_array, r2_array = [], []
t_array = []

# Initialize positions and velocities
r1, v1 = r1_0, v1_0
r2, v2 = r2_0, v2_0

# Main loop for adaptive RK4 integration
for t, r1, r2 in adaptive_rk4(r1, v1, r2, v2, t0, h0):
    t_array.append(t)
    r1_array.append(r1)
    r2_array.append(r2)

# Convert lists to arrays for plotting
r1_array = np.array(r1_array)
r2_array = np.array(r2_array)

# Plotting the trajectories
plt.figure(figsize=(8, 6))
plt.plot(r1_array[:, 0], r1_array[:, 1], label='Body 1')
plt.plot(r2_array[:, 0], r2_array[:, 1], label='Body 2')
plt.title('Trajectories of Two Bodies (Adaptive RK4 Method)')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.legend()
plt.grid
