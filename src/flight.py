import numpy as np
import util
# Constants
PIX2M = 2.1 # @17 zoom
D2R = np.pi/180.
GRAV = 100

def initial_state(X0 = 0., Y0 = 0., phi0 = 0., psi0 = 0.):
	state = (X0, Y0, phi0, psi0)
	return state

def next_state(state,state_dot,dt):
	# first order euler
	new_state = util.affine(state, state_dot, dt)
	X, Y, phi, psi = new_state
	if psi > 2. * np.pi:
		psi = np.mod(psi, 2.*np.pi)

	new_state = (X, Y, phi, psi)    

	return new_state

def plant(state, aircraft, psi_c):
	# unpack
	X,Y, phi, psi = state
	speed = aircraft['speed']
	max_bank = aircraft['max_bank']
	K_psi = aircraft['K_psi']
	K_phi = aircraft['K_phi']

	# velocity
	Vx = speed*np.cos(psi)
	Vy = speed*np.sin(psi)
	
	# lateral-directional 
	err_psi = psi_c - psi
	phi_c  = np.maximum(-max_bank,np.minimum(max_bank, K_psi*err_psi))
	err_phi = phi_c - phi
	dphi = K_phi * err_phi

	# turn dynamics
	dpsi = -GRAV*np.tan(phi)/speed
	
	state_dot = (Vx, Vy, dphi, dpsi)

	return state_dot

def followLine(curr_node, prev_node, next_node, weights):
	# A: position of ac
	# P: prev_node 
	# Q: next_node
	# R: closest point on line to position

	# weights balance going to the line versus keeping heading
	# [0] : get to node. [1] get to line.

	def vec2heading(vec):
		angle = np.arctan2(vec[1],vec[0])
		return np.mod(angle + 2.*np.pi, 2.*np.pi) 
	
	# points of interest
	A = np.array(curr_node)
	P = np.array(prev_node)
	Q = np.array(next_node)

	# command vector
	PQ = Q-P
	PA = A-P
	PR = PA.dot(PQ)/np.sum(PQ**2) * PQ 
	AR = PR - PA;
	
	# command psi
	psi_c = vec2heading(weights[0]*PQ + weights[1]*AR)
	
	if psi_c > np.pi:
		psi_c -= 2.*np.pi
	elif psi_c < -np.pi:
		psi_c += 2.*np.pi

	return psi_c







