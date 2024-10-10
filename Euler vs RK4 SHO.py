# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:57:38 2024

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation for the simple harmonic oscillator
def harmonic_oscillator(x, v, t, m, k):
    dxdt = v
    dvdt = -k/m * x
    return dxdt, dvdt

# Euler method for solving ordinary differential equations
def euler_method(harmonic_oscillator, y0, t, m, k):
    x, v = y0
    x_values = [x]
    v_values = [v]
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        dxdt, dvdt = harmonic_oscillator(x, v, t[i-1], m, k)
        x = x + dxdt * dt
        v = v + dvdt * dt
        x_values.append(x)
        v_values.append(v)
    
    return np.array(x_values), np.array(v_values)

# Runge-Kutta (RK4) method for solving ordinary differential equations
def runge_kutta_method(harmonic_oscillator, y0, t, m, k):
    x, v = y0
    x_values = [x]
    v_values = [v]
    
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        
        k1x, k1v = harmonic_oscillator(x, v, t[i-1], m, k)
        k2x, k2v = harmonic_oscillator(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, t[i-1] + 0.5 * dt, m, k)
        k3x, k3v = harmonic_oscillator(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, t[i-1] + 0.5 * dt, m, k)
        k4x, k4v = harmonic_oscillator(x + dt * k3x, v + dt * k3v, t[i-1] + dt, m, k)
        
        x = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x) * dt
        v = v + (1/6) * (k1v + 2*k2v + 2*k3v + k4v) * dt
        
        x_values.append(x)
        v_values.append(v)
    
    return np.array(x_values), np.array(v_values)

# Exact Solution of SHO
def f(y0,t, m, k):
    omega = np.sqrt(k / m)
    x0, v0 = y0
    
    # Analytic solution
    x_analytic = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)

    # Derivative of the analytic solution (velocity)
    v_analytic = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
    
    return  x_analytic, v_analytic

# Initial conditions
x0 = 1.0  # initial displacement
v0 = 0.0  # initial velocity
y0 = [x0, v0]

# Parameters
m = 1.0  # mass
k = 4.0  # spring constant

# Time points
t = np.linspace(0, 10.01,101)

# Solve the differential equation using Euler method
euler_sol = euler_method(harmonic_oscillator, y0, t, m, k)

# Solve the differential equation using RK4 method
rk4_sol = runge_kutta_method(harmonic_oscillator, y0, t, m, k)

# Exact Solution of SHO    
SHO_Sol =  f(y0,t, m, k)

# Plot the results
plt.plot(t, euler_sol[0], label='Euler', linestyle='--')
#plt.plot(t, euler_sol[1], label='Euler Velocity', linestyle='--')
plt.plot(t, rk4_sol[0], label='RK4')
#plt.plot(t, rk4_sol[1], label='RK4 Velocity')
plt.plot(t,SHO_Sol[0], label='Exact', linestyle='--')
#plt.plot(t,SHO_Sol[1], label='Exact Velocity', linestyle='--')
#plt.xlabel('Time (t)')
#plt.ylabel('Displacement x(t)')
plt.xlabel('t', fontsize=18)
plt.ylabel('x(t)', fontsize=18)
plt.title('Simple Harmonic Oscillator - Euler and RK4 Methods')
plt.legend()
plt.show()