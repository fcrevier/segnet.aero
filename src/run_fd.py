# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import numpy as np
import matplotlib.pyplot as plt
import flight_dynamics_simple as fd
# CONSTANTS
D2R = np.pi/180.

FPS = 20
dt = 1./float(FPS)
dt = 0.01
# Initial conditions 
npts = 1000
state = fd.initial_state()
state_history = np.zeros((npts,len(state)+1))
t = np.linspace(0., dt*npts, num = npts)
# objective
prev_node = (-10.,10.)
next_node = (20., 20.)


for ii in range(npts):
    # get state
    X, Y, phi, psi = state   

    # keep psi in [0 2pi)
    if psi > 2.*np.pi:
        psi = np.mod(psi,2.*np.pi)

    # compute body x vector (no theta)
    xb = (np.cos(psi), np.sin(psi), 0.)

    # TURN MANEUVER
    # controller
    speed = 10
    K_psi = -0.3
    K_phi = 3
    max_bank = 30.*D2R
    weights = (1.,2.) # (to_objective, to_line)

    # control
    psi_c = fd.followLine((X,Y), prev_node, next_node, weights)
    state_dot = fd.plant(state, speed, psi_c, K_psi, K_phi, max_bank)

    # update
    state = fd.next_state(state, state_dot, dt)
    state_history[ii,[0,1,2,3]] = np.array(state)
    state_history[ii,4] = psi_c

# PLOT HISTORY
# position
m = (next_node[1]-prev_node[1])/(next_node[0]-prev_node[0])
b = next_node[1] - m * next_node[0]
plt.plot(state_history[:,0],state_history[:,1],'k',state_history[:,0],b + m*state_history[:,0])
plt.show()

# attitude
plt.plot(t,state_history[:,[2,3,4]])
plt.show()