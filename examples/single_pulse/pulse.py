import numpy as np
import matplotlib.pyplot as plt
import classCompliance as Laplace
from matplotlib import animation
from scipy import interpolate
from numpy import where
from sys import exit

LNWDT=2; FNT=15
plt.rcParams['lines.linewidth'] = LNWDT; plt.rcParams['font.size'] = FNT

    
def inflow(t):
    return 10**-6*np.exp(-10000*(t-0.05)**2)
    
def UpdateRiemann(P, Q, A):
    
    U = Q/A
    C = AreaCompliance.C(P)
    c = np.sqrt(A/(rho*C))
    l1 = U*alpha + np.sqrt(c**2 + U**2*alpha*(alpha - 1))
    l2 = U*alpha - np.sqrt(c**2 + U**2*alpha*(alpha - 1))
    Z1 =  1.0/(C*l1)
    Z2 = -1.0/(C*l2)
    
    r11 = Z1
    r12 = Z2
    r21 = 1.0
    r22 = -1.0
    l11 =  1.0/(Z1+Z2)
    l12 =  Z2/(Z1+Z2)
    l21 =  1.0/(Z1+Z2)
    l22 = -Z1/(Z1+Z2)
    
    return l1, l2, r11, r12, r21, r22, l11, l12, l21, l22

        
def F(u):
    
    flux = np.zeros_like(u)
    p = u[0, :]
    q = u[1, :]
    A = AreaCompliance.A(p)
    C = AreaCompliance.C(p)
    flux[0, :] = q/C
    flux[1, :] = A*p/rho + alpha*q**2/A
    if np.min(A)<0:
        print "error, negative Area"
    if np.min(p)<-0.1:
        print "error, negative pressure: ", np.min(p)
    return flux 
    

def macCormack(u):
    up = u.copy()
    Utemp1 = u[1, :-1]/AreaCompliance.A(u[0, :-1])
    up[:, :-1] = u[:, :-1] - dt*(F(u[:, 1:]) - F(u[:, :-1]))/dx 
    up[1,:-1] = up[1, :-1] - dt*2*(gamma + 2)*my*Utemp1*np.pi/rho
    Utemp2 = up[1, 1:]/AreaCompliance.A(up[0, 1:])
    u[:, 1:] = .5*(u[:, 1:] + up[:, 1:] -  dt/dx*(F(up[:, 1:]) - F(up[:, :-1])))
    u[1, 1:] = u[1, 1:] - 0.5*dt*2*(gamma + 2)*my*Utemp2*np.pi/rho
    return u[:, 1:-1] 

### Main program

tmin, tmax = 0.0, 8.0 # start and stop time of simulation
xmin, xmax = 0.0, 10.0 # start and end of spatial domain
Nx = 3200 # number of spatial points
CFL = 0.99 # courant number, need CFL<=1 for stability

x = np.linspace(xmin, xmax, Nx+1) # discretization of space

#Parameters:

E = 400 *1e3
Pd = 0
Pext = 0
h = 1.5*1e-3
alpha = 1.
gamma = 9.0
rho = 1050.0
my = 0
viscous = False
fluxcorrection = False

if viscous:
    my = 4*10**-3
    
if fluxcorrection:
    alpha = (gamma + 2)/(gamma + 1)
    print "alpha: ", alpha
    
Ad = np.pi*10**-4
p0 = 1.0
AreaCompliance=Laplace.Laplace(E, h, Ad, Pext, Pd) #initialize Laplace Area and Compliance
C0 = AreaCompliance.C(0)
c = np.sqrt(Ad/(rho*C0))

dx = float((xmax-xmin)/Nx) # spatial step size
dt = round(CFL/c*dx,5) # stable time step calculated from stability requirement
Nt = int(round((tmax-tmin)/dt)) # number of time steps
time = np.linspace(tmin, tmax, Nt+1) # discretization of time

u = np.zeros((2, len(x)))
un = np.zeros((len(time), 2, len(x))) # holds the numerical solution, P and Q
un[0, 0, :] = np.ones_like(x)*0 #initial conditions for P (constant pressure)
un[0, 1, :] = np.ones_like(x)*0 #initial conditions for Q

reflectionCoefficient = 0.5
for i in range(1, Nt + 1):
    # calculate numerical solution of interior
    u[:,1:-1] = macCormack(un[i-1,:,:]) 
    
    # calculate inlet BC
    PinletLast = un[i - 1, 0, 0]
    QinletLast = un[i - 1, 1, 0]
    Ainletlast = AreaCompliance.A(PinletLast)
    l1, l2, r11, r12, r21, r22, l11, l12, l21, l22 = UpdateRiemann(PinletLast, QinletLast, Ainletlast) #caluculate Eigenvalues and Right eigenvalue matrix
    xint = x[0] - l2*dt
    Pint = np.interp(xint, x,un[i - 1, 0, :])
    Qint = np.interp(xint, x, un[i - 1, 1, :])
    dw2 = l21*(Pint - PinletLast) + l22*(Qint - QinletLast)
    dw1 = (inflow(time[i]) - QinletLast - r22*dw2)/r21
    
    
    dP = r11*dw1 + r12*dw2
    dQ = r21*dw1 + r22*dw2
    u[0, 0] = PinletLast + dP #(inflow(time[i])-Qint)*r11+Pint
    u[1, 0] = QinletLast + dQ #inflow(time[i])

    # calculate outlet BCw1
    PoutletLast = un[i - 1, 0, -1]
    QoutletLast = un[i - 1, 1, -1]
    Aoutletlast = AreaCompliance.A(PoutletLast)
    l1, l2, r11, r12, r21, r22, l11, l12, l21, l22 = UpdateRiemann(PoutletLast, QoutletLast, Aoutletlast) #caluculate Eigenvalues and Right eigenvalue matrix
    xint = x[-1] - l1*dt
    Pint = np.interp(xint, x, un[i - 1, 0, :])
    Qint = np.interp(xint, x, un[i - 1, 1, :])
    dw1 = l11*(Pint - PoutletLast) + l12*(Qint - QoutletLast)
    dw2 = dw1*reflectionCoefficient
    
    dP = r11*dw1 + r12*dw2
    dQ = r21*dw1 + r22*dw2
    
    u[0, -1] = PoutletLast + dP #np.interp(xint, x, un[i - 1, 0, :]) #calculate outlet boundary [P]from wavespeed and interpolate
    u[1, -1] = QoutletLast + dQ #np.interp(xint, x, un[i - 1, 1, :]) #calculate outlet boundary [Q]from wavespeed and interpolate
    
    
    un[i, :, :] = u[:, :] # storing the solution for plotting
    

### Animation 
# First set up the figure, the axis, and the plot element we want to animate

AnimatePulse=True

if AnimatePulse:
    
    fig = plt.figure()
    #title=plt.text(9,18,'time: ')
    ax = plt.axes(xlim=(0, 10), ylim=(-1, 1))
    line, = ax.plot([], [], lw=2)
    plt.xlabel('x-coordinate [m]')
    plt.ylabel('Flow [ml/s]')
    plt.tight_layout()
    
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        
        return line,
    
    # animation function.  This is called sequentially
    def animate(i):
    
        line.set_data(x, 1e6*un[i*jump, 1, :])
        #time_text.set_text('time = %.1f' % time[i*jump])
        return line  #,time_text
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    jump=20
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(time)/jump, interval=5, blit=False)
    plt.show()
