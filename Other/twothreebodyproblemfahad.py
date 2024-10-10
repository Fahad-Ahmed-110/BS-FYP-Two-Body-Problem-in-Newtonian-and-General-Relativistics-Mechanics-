# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:48:14 2024

@author: Hp
"""

import RK
from importlib import reload
reload(RK)
import numpy as np
import matplotlib.pyplot as plt

# Constants and Parameters
binary_separation = 20.0
mass_ratio = 1.0
M = 1.0
G = 1.0

# Dynamics Functions
def Kepler(t, y):
    """ Implements dX/dt = V, dV/dt = -M*G*X/|X|^3.
    """
    position = y[:3]
    velocity = y[3:]

    # Calculate position and velocity derivatives
    pos_dot = velocity
    r = np.linalg.norm(position)
    vel_dot = - position * M * G / (r ** 3)
    
    # Return derivatives
    return np.concatenate((pos_dot, vel_dot))

def Binary_position(t):
    """ Computes x, y, z coordinates of the two components of a binary in a circular orbit.
    """
    binary_frequency = np.sqrt(M / binary_separation ** 3)
    binary_angle = t * binary_frequency
    X1 = binary_separation * np.array([np.cos(binary_angle), np.sin(binary_angle), 0]) / (1.0 + mass_ratio)
    X2 = -X1 
    return X1, X2

def Binary(t, y):
    """ Source term for a binary. Assumes circular orbits.
    """
    m1 = M * mass_ratio / (1. + mass_ratio)
    m2 = M / (1. + mass_ratio)

    X1, X2 = Binary_position(t)
    position = y[:3]
    velocity = y[3:]

    # Calculate position and velocity derivatives
    pos_dot = velocity
    p1 = position - X1
    p2 = position - X2
    r1 = np.linalg.norm(p1)
    r2 = np.linalg.norm(p2)
    vel_dot = - p1 * m1 * G / (r1 ** 3) - p2 * m2 * G / (r2 ** 3)
    
    # Return derivatives
    return np.concatenate((pos_dot, vel_dot))

# Initial Conditions
x0, y0, z0 = 40.0, 0.0, 0.0
pert = 0.0 
Vx, Vy, Vz = 0.0, np.sqrt(1.0 / x0) + pert, 0.0
y_initial = np.array([x0, y0, z0, Vx, Vy, Vz], dtype=np.float64)
t, tend, dt = 0, 20000, 0.1

# Plotting
def plot_output(t, y):
    plt.plot(y[0], y[1], color='green', marker='o', linestyle='dashed', markersize=5)
    BH1, BH2 = Binary_position(t)
    plt.plot(BH1[0], BH1[1], color='black', marker='o', linestyle='dashed', markersize=4)
    plt.plot(BH2[0], BH2[1], color='orange', marker='o', linestyle='dashed', markersize=4)
    plt.clf()

# Simulation Loop
delta_t_dump = 1
t_dump = 0
plot_output(t, y_initial)

while t < tend:
    told = t
    t, y_initial, dt = RK.RK45_Step(t, y_initial, dt, Binary, tol=1.0e-12)
    if t >= t_dump + delta_t_dump:
        plot_output(t, y_initial)
        t_dump = t

plt.show()
