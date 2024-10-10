# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:44:32 2024

@author: Fahad
"""

# Finding a Stable Circular Orbit with Fixed Suns
    
import numpy as np
import matplotlib.pyplot as plt

# Define global parameters
M = 1.0  # Mass of one of the bodies
binary_separation = 1.0  # Separation of the binary system
mass_ratio = 1.0  # Ratio of the masses (for simplicity, set to 1.0)

def Binary_position(t):
    """ Gives x,y,z coordinated of the two components of a binary in a circular orbit. """
    binary_frequency = np.sqrt(M / binary_separation**3)
    binary_angle = t * binary_frequency
    xsun1 = binary_separation * np.array([np.cos(binary_angle), np.sin(binary_angle)]) / (1.0 + mass_ratio)
    xsun2 = -xsun1 
    return xsun1, xsun2

# Define the differential equations for the two-body system
def two_body_system(x, v, t):
    G = 1.0  # Gravitational constant
    xsun1, xsun2 = Binary_position(t)  # Update xsun1 and xsun2 using Binary_position function
    r1 = np.sqrt(np.dot(x-xsun1, x-xsun1))  # Distance from the first sun
    r2 = np.sqrt(np.dot(x-xsun2, x-xsun2))  # Distance from the second sun

    dxdt = v
    dvdt = -G * (M / r1**3 * (x - xsun1) + M / r2**3 * (x - xsun2))
    
    return dxdt, dvdt

# Function to compute the velocity for circular orbit at a given radius
def circular_velocity(radius):
    G = 1.0  # Gravitational constant
    velocity = np.sqrt(G * M / radius)
    return np.array([0.0, velocity])

# Compute the desired radius for the circular orbit
desired_radius = 2.0

# Compute the initial position and velocity vectors for the circular orbit
x0 = np.array([desired_radius, 0.0])
v0 = circular_velocity(desired_radius)
y0 = [x0, v0]

# Time points
t = np.arange(0, 50.0, 0.1)

# Runge-Kutta (RK4) method for solving ordinary differential equations
def runge_kutta_method(two_body_system, y0, t):
    x, v = y0
    x_values = [x]
    v_values = [v]
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        
        k1x, k1v = two_body_system(x, v, t[i-1])
        k2x, k2v = two_body_system(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t[i-1] + 0.5 * dt)
        k3x, k3v = two_body_system(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t[i-1] + 0.5 * dt)
        k4x, k4v = two_body_system(x + dt * k3x, v + dt * k3v, t[i-1] + dt)
        
        x = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x) * dt
        v = v + (1/6) * (k1v + 2*k2v + 2*k3v + k4v) * dt
        
        x_values.append(x)
        v_values.append(v)
        
    return np.array(x_values), np.array(v_values)

# Solve the differential equation using RK4 method
rk4_solution = runge_kutta_method(two_body_system, y0, t)

# Plotting the results
plt.figure(figsize=(8, 6))

# Position plot
xsun1, xsun2 = Binary_position(t[-1])  # Compute xsun1 and xsun2 at the final time point
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
