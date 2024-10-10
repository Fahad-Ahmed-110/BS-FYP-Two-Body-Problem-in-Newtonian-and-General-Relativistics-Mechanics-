#This file is to compute all the geodesics around Kerr spacetime.
#Jam Sadiq Aug 8th, 2016
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pow

#set in the start of code numerical spacing
h_finite_diff = 1e-3 #accuracy for finite difference derivs
spin_kerr = 0.5 # 0.0 => Schwarzschild ; for Kerr a can go 0 < a < 1
M = 1.0  #mass of BH
dt = 0.5  #for integration step
tend = 2000  #how long we want this run
#Initial Position and Velocity of orbiting body
#Initial Conditions  need same as Cartesian x0 = r , y0 = theta z0 = phi 
#We are using spherical coords here so we need translation to be in Cartesian
t0 = 0.0
R0 = 6.0        #geodesics at this distance from BHs 
Thta0 = np.pi/2.0  #mean we are on equator
Phi0 = 0.0        #We want velocity to body in phi direction
Rdot0 = 0.0            # vr
Thtadot0 = 0.0  #vtheta theta=90degree geodesics are on equator of sphere
Phidot0 = 1.0/(R0*sqrt(R0 -3))  # Worked out for circular orbit of schwarzschild

# Kerr Metric
def get_Kerrgab(t, r, theta, phi, a=0.0):
    """
    Kerr metric in Boyer-Lindquist Coordinates
    check in https://en.wikipedia.org/wiki/Kerr_metric
    a = 0.0 => Schwarzschild mertic by default
    """
    r2 = r*r
    a2 = a*a
    rho2 = r2 + a2*cos(theta)*cos(theta)
    Del  = r2 -2*M*r +a2
    Sigma = (r2 +a2)* (r2 +a2)  - a2*Del*sin(theta)*sin(theta)
    g00 =  -1+ (2*M*r)/rho2
    g01 = 0
    g02 = 0
    g03 = -2*M*a*r*sin(theta)*sin(theta)/rho2
    g11 = rho2/Del
    g12 = 0
    g13 = 0
    g22 = rho2
    g23 = 0
    g33 = Sigma*sin(theta)*sin(theta)/rho2
  
    return np.array(((g00, g01, g02, g03), (g01, g11, g12, g13), (g02, g12, g22, g23),(g03, g13, g23, g33)), dtype=np.float64)
  
#Geodesic Equations require Gamma or christofell symbols
# we need derivatives of metric
#Extra Higher order derivatives
#=======================================================

def tderiv(f,t,x,y,z):
    h = h_finite_diff
    return (-f(t+2*h,x,y,z)+ 8*f(t+h,x,y,z)-8*f(t-h,x,y,z)+f(t-2*h,x,y,z))/(12*h)
def xderiv(f,t,x,y,z):
    h = h_finite_diff
    return (-f(t,x+2*h,y,z)+ 8*f(t,x+h,y,z)-8*f(t,x-h,y,z)+f(t,x-2*h,y,z))/(12*h)
def yderiv(f,t,x,y,z):
    h = h_finite_diff
    return  (-f(t,x,y+2*h,z)+ 8*f(t,x,y+h,z)-8*f(t,x,y-h,z)+f(t,x,y-2*h,z))/(12*h)
def zderiv(f,t,x,y,z):
    h = h_finite_diff
    return  (-f(t,x,y,z+2*h)+ 8*f(t,x,y,z+h)-8*f(t,x,y,z-h)+f(t,x,y,z-2*h))/(12*h)
#=========================================================
#Christoffel connection symbols 
def get_christoffel(t, x, y, z, get_gab, a=0.0):
    """
    get_gab will be get_Kerrgab(t, r, theta, phi, a=0.0)
    """
    t=np.float64(t)
    x =np.float64(x)
    y = np.float64(y)
    z = np.float64(z)
    
    gab = get_gab(t, x, y, z, a=a)
    inverse_gab =  np.linalg.inv(gab)
    
    
    allgabderiv=[0,0,0,0]
    allgabderiv[0] = tderiv(get_gab, t,x,y,z )
    allgabderiv[1] = xderiv(get_gab, t,x,y,z )
    allgabderiv[2] = yderiv(get_gab, t,x,y,z )
    allgabderiv[3] = zderiv(get_gab, t,x,y,z )
    
    christoffel = np.ndarray(shape=(4,4,4), dtype=np.float64)
    
    for c in range(4):
        for a in range(4):
            for b in range(4):
                christoffel[c, a, b] = 0
                for d in range(4):
                    christoffel[c,a,b] += 0.5 * inverse_gab[c, d]*\
                        (allgabderiv[a][b,d] + allgabderiv[b][a,d] -\
                         allgabderiv[d][a,b])
    return christoffel

################## Part II Numerical Integration RK4, RK45step ########
default_tol = 1.0e-10
def RK4_Step(xold, yold, dx, fxy):
  k1 = fxy(xold, yold)
  k2 = fxy(xold+ 0.5 * dx, yold + 0.5 * dx * k1)
  k3 = fxy(xold+ 0.5 * dx, yold + 0.5 * dx * k2)
  k4 = fxy(xold+ 1.0 * dx, yold + 1.0 * dx * k3)

  xnew = xold + dx
  ynew = yold + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
  return xnew, ynew, dx

# Adaptive RK45
def RK45_Step(xold, yold, dx, fxy, tol=default_tol):
  k1 = dx * fxy(xold, yold)
  k2 = dx * fxy(xold + 1.0/4.0 * dx, yold + 1.0/4.0 * k1)
  k3 = dx * fxy(xold + 3.0/8.0 * dx, yold + 3.0/32.0 * k1 + 9.0/32.0 * k2)
  k4 = dx * fxy(xold + 12.0/13.0 * dx, yold + 1932.0/2197.0 * k1 -\
                 7200.0/2197.0 * k2 + 7296.0/2197.0 * k3)
  k5 = dx * fxy(xold + dx, yold + 439.0/216.0 * k1 - 8.0 * k2 +\
                 3680.0/513.0 * k3 - 845.0/4104.0 * k4)
  k6 = dx * fxy(xold + 1.0/2.0 * dx, yold -8.0/27.0 * k1 + 2.0 * k2 -\
                 3544.0/2565.0 * k3 + 1859.0/4104.0 * k4 - 11.0/40.0 * k5)

  # higher order accuracy ynew
  ynew = yold + 16.0/135.0 * k1 + 6656.0/12825.0 * k3 +\
             28561.0/56430.0 * k4 -9.0/50.0 * k5 + 2.0/55.0 * k6

  # lower order accuracy ynew
  ynew4 = yold + 25.0/216.0 * k1 + 1408.0/2565.0 * k3 +\
             2197.0/4104.0 * k4 -1.0/5.0 * k5

  xnew = xold + dx

  err= abs(np.array(ynew-ynew4)).max()

#  if err < tol * 1.0e-3:
#    dtc = 2* dt
#  else:

  dxnew = 0.9 * dx * (tol/err)**(0.25)

  if err > tol:
    ynew = yold
    xnew = xold

  return (xnew, ynew, dxnew)

############################### Experiment #######################
M = 1.0  #mass of BH
a = 0.9 #a = 0.0 => Schwarzschild ; for Kerr a can go 0 < a < 1
dt = 0.5 #for integration step
tend = 2000  #how long we want this run
#Initial Position and Velocity of orbiting body
#Initial Conditions  need same as Cartesian x0 = r , y0 = theta z0 = phi 
#We are using spherical coords here so we need translation to be in Cartesian

t0 = 0.0
R0 = 6.0        #geodesics at this distance from BHs 
Thta0 = np.pi/2.0  #mean we are on equator
Phi0 = 0.0        #We want velocity to body in phi direction
Rdot0 = 0.0            # vr
Thtadot0 = 0.0  #vtheta theta=90degree geodesics are on equator of sphere
Phidot0 = 1.0/(R0*sqrt(R0 -3))  # Worked out for circular orbit

gab = get_Kerrgab(t0, R0, Thta0, Phi0, a=spin_kerr) 
#For timelike geodesics initial velocity components
Uup = np.array((1, Rdot0, Thtadot0, Phidot0), dtype=np.float64)
#For timelike geodesics  Ua Ub gab==-1  #for  light-like geodesic ==0
## solving equation -ut^2 gtt + ur^2 grr  == -1 [solve for utt] complicated for Kerr
Aa = gab[0,0]
Bb = 0
Cc = 0
for i in range(1,4):
  Bb = Bb + 2 *gab[0,i] * Uup[i]
  for j in range(1,4):
    Cc = Cc + gab[i,j]* Uup[i]*Uup[j]
Cc = Cc+1
Dd =np.sqrt(Bb**2 - 4*Aa*Cc)
Uup[0] = (-Bb - Dd)/(2*Aa) 
print("check it is correct root +ve one for ut0 = ", Uup[0])

def RHS(time, Svec):
    """
    Geodesic Equations in first derivative
    dv^{a}/dt = -Gamma^{a}_{bc}v^{b}v^{c}
    """
    Tt    = Svec[0]   #t,x,y,z
    Rr    = Svec[1]
    Tthta = Svec[2]
    Pphi  = Svec[3]
    u = Svec[4:8]   #ut,ux,uy,uz
    
    Gamma = get_christoffel(Tt,Rr,Tthta,Pphi, get_Kerrgab, a=spin_kerr)
    out = np.ndarray(shape = 8, dtype = np.float64)
    out[0] = u[0]   #Tdot
    out[1] = u[1]   #Rdot
    out[2] = u[2]   #Thtadot
    out[3] = u[3]   #Phidot
    
    for c in range(4):
        out[4+c] = 0  #setting zero and then filling in the geodesic equation
        for a in range(4):
            for b in range(4):
                out[4+c] -= Gamma[c, a, b]*u[a]*u[b]
    return out

##===============Initial data
yold  = np.ndarray(shape = 8, dtype = np.float64)   # i change shape=8 to 20
yold[0] = t0        #t0
yold[1] = R0        #x0
yold[2] = Thta0     #y0
yold[3] = Phi0      #z0
yold[4] = Uup[0]    # we need condition on it gab ua ub == -1
yold[5] = Uup[1]    #ux0
yold[6] = Uup[2]    #uy0  using gm/r = v**2/r, pow
yold[7] = Uup[3]    #uz0

t = 0.0
TArray =[t]
XArray=[R0]
YArray=[Phi0]

#############solve via RK4 or RK45
while  t < tend:
  (t, yold, dt) = RK4_Step(t, yold, dt, RHS)#, tol=1.0e-10)
  #(t, yold, dt) = RK45_Step(t, yold, dt, RHS, tol=1.0e-10)
  TArray.append(t)
  XArray.append(yold[1]) #r
  YArray.append(yold[3]) #phi
  
#plot orbit: x = rcos(phi),y = rsin(phi) positions notice theta=
plt.figure(figsize=(8,8))
plt.plot(XArray*np.cos(YArray), XArray*np.sin(YArray), 'r', label = "R = 6M, a={0}".format(a))
plt.scatter(0, 0, c='k', s=20)
plt.legend(loc = 1, fontsize = 16)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
