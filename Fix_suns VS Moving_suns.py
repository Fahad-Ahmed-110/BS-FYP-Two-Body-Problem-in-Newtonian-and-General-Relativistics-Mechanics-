# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:46:21 2024

@author: Fahad
"""

# Comparison of Fixed Sun vs. Moving Sun Initial Conditions

import numpy as np
import matplotlib.pyplot as plt

# Define global parameters
M = 1.0  # Mass of one of the bodies
binary_separation = 1.0  # Separation of the binary system
mass_ratio = 1.0  # Ratio of the masses (for simplicity, set to 1.0)

def Binary_position(t, n):
    """ Gives x,y,z coordinated of the two components of a binary in a circular orbit. """
    binary_frequency = np.sqrt(M / binary_separation**3)
    binary_angle = t * binary_frequency
    xsun1 = np.zeros((2, n))  # Initialize xsun1 array
    xsun2 = np.zeros((2, n))  # Initialize xsun2 array
    for i in range(n):
        xsun1[:, i] = binary_separation * np.array([np.cos(binary_angle[i]), np.sin(binary_angle[i])]) / (1.0 + mass_ratio)
        xsun2[:, i] = -xsun1[:, i]
    return xsun1, xsun2

# Define the differential equations for the two-body system
def two_body_system(x, v, t, xsun1, xsun2):
    G = 1.0  # Gravitational constant
    r1 = np.sqrt(np.sum((x - xsun1)**2, axis=0))  # Distance from the first sun
    r2 = np.sqrt(np.sum((x - xsun2)**2, axis=0))  # Distance from the second sun

    dxdt = v
    dvdt = -G * (M / r1**3 * (x - xsun1) + M / r2**3 * (x - xsun2))
    return dxdt, dvdt

# Runge-Kutta (RK4) method for solving ordinary differential equations
def runge_kutta_method(two_body_system, y0, t, xsun1, xsun2):
    x, v = y0
    x_values = [x]
    v_values = [v]
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        
        k1x, k1v = two_body_system(x, v, t[i-1], xsun1[:, i-1], xsun2[:, i-1])
        k2x, k2v = two_body_system(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t[i-1] + 0.5 * dt, xsun1[:, i-1], xsun2[:, i-1])
        k3x, k3v = two_body_system(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t[i-1] + 0.5 * dt, xsun1[:, i-1], xsun2[:, i-1])
        k4x, k4v = two_body_system(x + dt * k3x, v + dt * k3v, t[i-1] + dt, xsun1[:, i-1], xsun2[:, i-1])
        
        x = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x) * dt
        v = v + (1/6) * (k1v + 2*k2v + 2*k3v + k4v) * dt
        
        x_values.append(x)
        v_values.append(v)
        
    return np.array(x_values), np.array(v_values)

# Initial conditions
x0 = np.array([18.5, 0.0])  # initial position vector
v0 = np.array([0.0, 0.25])  # initial velocity vector
y0 = [x0, v0]

# Time points
t = np.arange(0, 300.0, 0.1)

# Compute positions of the suns over time
n = len(t)
xsun1, xsun2 = Binary_position(t, n)

# Solve the differential equation using RK4 method
rk4_solution = runge_kutta_method(two_body_system, y0, t, xsun1, xsun2)

# Plotting the results
plt.figure(figsize=(8, 6))


# Position plot
plt.plot(xsun1[0], xsun1[1], 'ro')  # Plotting the first sun
plt.plot(xsun2[0], xsun2[1], 'ro')  # Plotting the second sun
plt.plot(rk4_solution[0][:, 0], rk4_solution[0][:, 1], label='RK4', color='blue', linestyle='-')
plt.title('Position (x-y)')
plt.xlabel('Position (x)')
plt.ylabel('Position (y)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
