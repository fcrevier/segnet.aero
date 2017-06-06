# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import numpy as np
import matplotlib.pyplot as plt
import flight_dynamics as fd

# CONSTANTS
D2R = np.pi/180.


dt = 0.01
# Initial conditions 
npts = 1000
state = fd.initial_state()
state_history = np.zeros((npts,len(state)))
t = np.linspace(0., dt*npts, num = npts)
# objective
prev_node = (0.,8.)
next_node = (20., 20.)

# controller
K_psi = -1
K_phi = .1
max_bank = 30.*D2R
weights = (1.,4.)

# loop
for ii in range(npts):

    # get state
    X, Y, Z, phi, theta, psi = state[:6]    

    # keep psi in [0 2pi)
    if psi > 2.*np.pi:
        psi = np.mod(psi,2.*np.pi)

    # compute body x vector (no theta)
    xb = (np.cos(psi), np.sin(psi), 0.)
    

    # TURN MANEUVER

    u_phi = fd.follow_line(state,
                        prev_node, next_node, weights, 
                        K_psi, K_phi, max_bank)
    
    # plant dynamics
    Ixx = 1.
    damp_roll = 1.
    state_dot = fd.turn(state, u_phi, Ixx, damp_roll)

    # update
    state = fd.next_state(state, state_dot, dt)
    state_history[ii,:] = np.array(state)


# PLOT HISTORY
plt.plot(t,state_history[:,[3,5]])
plt.show()
plt.plot(state_history[:,0],state_history[:,1])
plt.show()