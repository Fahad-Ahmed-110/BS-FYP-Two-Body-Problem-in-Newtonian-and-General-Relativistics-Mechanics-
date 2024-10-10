# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 17:49:46 2024

@author: Fahad
"""

# Three Body Problem by_RK4:

import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)

# Initial conditions
m1 = m2 = m3 = 1.0  # masses of all bodies (kg)
t0 = 0.0  # initial time
tf = 10000.0  # final time
dt = 0.1  # time step size

# Initial positions and velocities
r1_0 = np.array([1.0, 0.0])  # initial position of body 1 (m)
v1_0 = np.array([0.0, 1.0])  # initial velocity of body 1 (m/s)
r2_0 = np.array([-1.0, 0.0])  # initial position of body 2 (m)
v2_0 = np.array([0.0, -1.0])  # initial velocity of body 2 (m/s)
r3_0 = np.array([0.0, 0.0])  # initial position of body 3 (m)
v3_0 = np.array([0.0, 0.0])  # initial velocity of body 3 (m/s)

# Function to compute acceleration (force) on each body
def compute_acceleration(r1, r2, r3):
    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2
    
    a1 = G * (m2 / np.linalg.norm(r12)**3 * r12 + m3 / np.linalg.norm(r13)**3 * r13)
    a2 = G * (-m1 / np.linalg.norm(r12)**3 * r12 + m3 / np.linalg.norm(r23)**3 * r23)
    a3 = G * (-m1 / np.linalg.norm(r13)**3 * r13 - m2 / np.linalg.norm(r23)**3 * r23)
    
    return a1, a2, a3

# RK4 integration step
def rk4_step(r1, v1, r2, v2, r3, v3, dt):
    k1v1 = dt * compute_acceleration(r1, r2, r3)[0]
    k1v2 = dt * compute_acceleration(r2, r1, r3)[1]
    k1v3 = dt * compute_acceleration(r3, r1, r2)[2]
    
    k1r1 = dt * v1
    k1r2 = dt * v2
    k1r3 = dt * v3

    k2v1 = dt * compute_acceleration(r1 + 0.5 * k1r1, r2 + 0.5 * k1r2, r3 + 0.5 * k1r3)[0]
    k2v2 = dt * compute_acceleration(r2 + 0.5 * k1r2, r1 + 0.5 * k1r1, r3 + 0.5 * k1r3)[1]
    k2v3 = dt * compute_acceleration(r3 + 0.5 * k1r3, r1 + 0.5 * k1r1, r2 + 0.5 * k1r2)[2]
    
    k2r1 = dt * (v1 + 0.5 * k1v1)
    k2r2 = dt * (v2 + 0.5 * k1v2)
    k2r3 = dt * (v3 + 0.5 * k1v3)

    k3v1 = dt * compute_acceleration(r1 + 0.5 * k2r1, r2 + 0.5 * k2r2, r3 + 0.5 * k2r3)[0]
    k3v2 = dt * compute_acceleration(r2 + 0.5 * k2r2, r1 + 0.5 * k2r1, r3 + 0.5 * k2r3)[1]
    k3v3 = dt * compute_acceleration(r3 + 0.5 * k2r3, r1 + 0.5 * k2r1, r2 + 0.5 * k2r2)[2]
    
    k3r1 = dt * (v1 + 0.5 * k2v1)
    k3r2 = dt * (v2 + 0.5 * k2v2)
    k3r3 = dt * (v3 + 0.5 * k2v3)

    k4v1 = dt * compute_acceleration(r1 + k3r1, r2 + k3r2, r3 + k3r3)[0]
    k4v2 = dt * compute_acceleration(r2 + k3r2, r1 + k3r1, r3 + k3r3)[1]
    k4v3 = dt * compute_acceleration(r3 + k3r3, r1 + k3r1, r2 + k3r2)[2]
    
    k4r1 = dt * (v1 + k3v1)
    k4r2 = dt * (v2    + k3v2)
    k4r3 = dt * (v3 + k3v3)

    r1_new = r1 + (k1r1 + 2 * k2r1 + 2 * k3r1 + k4r1) / 6
    v1_new = v1 + (k1v1 + 2 * k2v1 + 2 * k3v1 + k4v1) / 6
    r2_new = r2 + (k1r2 + 2 * k2r2 + 2 * k3r2 + k4r2) / 6
    v2_new = v2 + (k1v2 + 2 * k2v2 + 2 * k3v2 + k4v2) / 6
    r3_new = r3 + (k1r3 + 2 * k2r3 + 2 * k3r3 + k4r3) / 6
    v3_new = v3 + (k1v3 + 2 * k2v3 + 2 * k3v3 + k4v3) / 6

    return r1_new, v1_new, r2_new, v2_new, r3_new, v3_new

# Arrays to store positions over time
time_steps = int((tf - t0) / dt)
t_array = np.linspace(t0, tf, time_steps)
r1_array = np.zeros((time_steps, 2))
r2_array = np.zeros((time_steps, 2))
r3_array = np.zeros((time_steps, 2))

# Initialize positions and velocities
r1, v1 = r1_0, v1_0
r2, v2 = r2_0, v2_0
r3, v3 = r3_0, v3_0

# Main loop for RK4 integration
for i in range(time_steps):
    r1_array[i] = r1
    r2_array[i] = r2
    r3_array[i] = r3
    
    r1, v1, r2, v2, r3, v3 = rk4_step(r1, v1, r2, v2, r3, v3, dt)

# Plotting the trajectories
plt.figure(figsize=(8, 6))
plt.plot(r1_array[:, 0], r1_array[:, 1], label='Body 1')
plt.plot(r2_array[:, 0], r2_array[:, 1], label='Body 2')
plt.plot(r3_array[:, 0], r3_array[:, 1], label='Body 3')
plt.title('Trajectories of Three Bodies (RK4 Method)')
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.legend()
plt.grid(True)
plt.show()

