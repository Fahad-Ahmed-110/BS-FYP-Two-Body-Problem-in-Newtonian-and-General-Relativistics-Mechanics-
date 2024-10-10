# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:20:13 2024

@author: Fahad

"""
import numpy as np
import matplotlib.pyplot as plt

# Define the gravitational constant
G = 1.0

# Define the equations of motion for the two-body system
def two_body(x, v):
    r = np.sqrt(x**2)
    dxdt = v
    dvdt = -G * x / r**3
    return dxdt, dvdt

# Runge-Kutta (RK4) method for solving ordinary differential equations
def runge_kutta_method(two_body, y0, t):
    x, v = y0
    x_values = [x]
    v_values = [v]
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        
        k1x, k1v = two_body(x, v)
        k2x, k2v = two_body(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = two_body(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = two_body(x + dt * k3x, v + dt * k3v)
        
        x = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x) * dt
        v = v + (1/6) * (k1v + 2*k2v + 2*k3v + k4v) * dt
        
        x_values.append(x)
        v_values.append(v)
    
    return np.array(x_values), np.array(v_values)

# Initial conditions
x0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity
y0 = [x0, v0]

# Time points
t = np.linspace(0, 10, 1001)

# Solve the differential equation using RK4 method
solution = runge_kutta_method(two_body, y0, t)

# Plotting the results
plt.figure(figsize=(8, 6))

# Position plots
plt.plot(t, solution[0], label='Position (x)', color='Blue')
plt.title('Position vs Time')
plt.xlabel('Time')
plt.ylabel('Position (x)')
plt.legend()
plt.grid(True)

plt.show()
