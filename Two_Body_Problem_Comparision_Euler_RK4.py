# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:12:45 2024

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt

# Exact solution for the two-body problem (circular orbit assumption)
def exact_solution(t, x0, v0):
    r = np.linalg.norm(x0)
    omega = np.linalg.norm(v0) / r  # Angular velocity for a circular orbit
    x_exact = r * np.cos(omega * t)
    y_exact = r * np.sin(omega * t)
    return np.vstack((x_exact, y_exact)).T

# Euler method for solving the two-body problem
def euler_method(two_body_system, y0, t):
    x, v = y0
    x_values = [x]
    v_values = [v]
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        
        dxdt, dvdt = two_body_system(x, v, t[i-1])
        
        x = x + dxdt * dt
        v = v + dvdt * dt
        
        x_values.append(x)
        v_values.append(v)
        
    return np.array(x_values), np.array(v_values)

# The given Runge-Kutta (RK4) method for solving the two-body problem
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

# Define the differential equations for the two-body system
def two_body_system(x, v, t):
    G = 1.0  # Gravitational constant
    M = 1.0  # Mass of one of the bodies 
    r = np.sqrt(np.dot(x-xsun,x-xsun))  # Distance between the two bodies

    dxdt = v
    dvdt = -G * M / r**(3/2) * (x-xsun)
    
    return dxdt, dvdt

# Initial conditions
x0 = np.array([2.0, 0.0])  # initial position vector
v0 = np.array([0.0, 1.0])  # initial velocity vector
y0 = [x0,v0]

# Time points
t = np.arange(0, 100.0, 0.1)

# Position of the fixed body (sun)
xsun = np.array([1, 0])

# Solve the differential equation using exact solution
exact_sol = exact_solution(t, x0, v0)

# Solve the differential equation using Euler method
euler_solution = euler_method(two_body_system, y0, t)

# Solve the differential equation using RK4 method
rk4_solution = runge_kutta_method(two_body_system, y0, t)

# Plotting the results
plt.figure(figsize=(10, 8))

# Exact solution plot
plt.plot(exact_sol[:, 0], exact_sol[:, 1], label='Exact Solution', color='green', linestyle='-')

# Euler method plot
plt.plot(euler_solution[0][:, 0], euler_solution[0][:, 1], label='Euler Method', color='red', linestyle='--')

# RK4 method plot
plt.plot(rk4_solution[0][:, 0], rk4_solution[0][:, 1], label='RK4 Method', color='blue', linestyle='-.')

# Position of the sun
plt.plot(xsun[0], xsun[1], 'ro', label='Fixed Body (Sun)')

plt.title('Comparison of Exact, Euler, and RK4 Methods')
plt.xlabel('Position (x)')
plt.ylabel('Position (y)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
