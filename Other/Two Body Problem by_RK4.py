# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:45:12 2024

@author: Fahad
"""

# Two Body Problem by RK4:

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)

# Initial conditions
m1 = m2 = 1.0  # masses of both bodies (kg)
t0 = 0.0  # initial time
tf = 10000.0  # final time
dt = 0.1  # time step size

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
def rk4_step(r1, v1, r2, v2, dt):
    k1v1 = dt * compute_acceleration(r1, r2)[0]
    k1v2 = dt * compute_acceleration(r2, r1)[1]
    k1r1 = dt * v1
    k1r2 = dt * v2

    k2v1 = dt * compute_acceleration(r1 + 0.5 * k1r1, r2 + 0.5 * k1r2)[0]
    k2v2 = dt * compute_acceleration(r2 + 0.5 * k1r2, r1 + 0.5 * k1r1)[1]
    k2r1 = dt * (v1 + 0.5 * k1v1)
    k2r2 = dt * (v2 + 0.5 * k1v2)

    k3v1 = dt * compute_acceleration(r1 + 0.5 * k2r1, r2 + 0.5 * k2r2)[0]
    k3v2 = dt * compute_acceleration(r2 + 0.5 * k2r2, r1 + 0.5 * k2r1)[1]
    k3r1 = dt * (v1 + 0.5 * k2v1)
    k3r2 = dt * (v2 + 0.5 * k2v2)

    k4v1 = dt * compute_acceleration(r1 + k3r1, r2 + k3r2)[0]
    k4v2 = dt * compute_acceleration(r2 + k3r2, r1 + k3r1)[1]
    k4r1 = dt * (v1 + k3v1)
    k4r2 = dt * (v2 + k3v2)

    r1_new = r1 + (k1r1 + 2 * k2r1 + 2 * k3r1 + k4r1) / 6
    v1_new = v1 + (k1v1 + 2 * k2v1 + 2 * k3v1 + k4v1) / 6
    r2_new = r2 + (k1r2 + 2 * k2r2 + 2 * k3r2 + k4r2) / 6
    v2_new = v2 + (k1v2 + 2 * k2v2 + 2 * k3v2 + k4v2) / 6

    return r1_new, v1_new, r2_new, v2_new

# Arrays to store positions over time
time_steps = int((tf - t0) / dt)
t_array = np.linspace(t0, tf, time_steps)
r1_array = np.zeros((time_steps, 2))
r2_array = np.zeros((time_steps, 2))

# Initialize positions and velocities
r1, v1 = r1_0, v1_0
r2, v2 = r2_0, v2_0

# Main loop for RK4 integration
for i in range(time_steps):
    r1_array[i] = r1
    r2_array[i] = r2
    r1, v1, r2, v2 = rk4_step(r1, v1, r2, v2, dt)

# Plotting the trajectories
plt.figure(figsize=(8, 6))
plt.plot(r1_array[:, 0], r1_array[:, 1], label='Body 1')
plt.plot(r2_array[:, 0], r2_array[:, 1], label='Body 2')
plt.title('Trajectories of Two Bodies (RK4 Method)')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.legend()
plt.grid(True)
plt.show()
