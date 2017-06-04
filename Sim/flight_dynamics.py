import numpy as np

# Constants
PIX2M = 2.1 # @17 zoom
D2R = np.pi/180.
GRAV = 9.8
def pqr(euler, euler_dot):
	phi, theta, psi = euler
	L = np.array([[1., 0., -np.sin(theta)],
		[0., np.cos(phi), np.sin(phi)*np.cos(theta)],
		[0., -np.sin(phi), np.cos(phi)*np.cos(theta)]])
	return L.dot(np.array(euler_dot))

def affine(t1, t2, alpha):
	if len(t1) != len(t2): 
		print 'error, tuples not of same length'

	return tuple(t1[i]+alpha*t2[i] for i in range(len(t1)))

def next_state(state,state_dot,dt):
	# first order euler
	new_state = affine(state, state_dot,dt)
	return new_state

def initial_state():

	XYZ = (0.,0.,0.)
	uvw = (30., 0.,0.)
	euler = (0., 0., 0)
	euler_dot = (0., 0., 0)
	state = XYZ + euler + uvw + euler_dot
	return state

def turn(state, u_phi, Ixx):
	# unpack
	X,Y,Z,phi,theta, psi,u,v,w, dphi, dtheta, dpsi = state
	# velocity
	V = np.array((u,v,w))
	Vnorm = np.linalg.norm(V)
	Vx = Vnorm*np.cos(psi)
	Vy = Vnorm*np.sin(psi)

	# turn dynamics	
	dpsi = GRAV*np.tan(phi)/Vnorm
	state_dot = (Vx, Vy, 0.,
		dphi, 0., dpsi, 
		0., 0., 0., 		
		u_phi/Ixx, 0., 0.)

	return state_dot

def follow_line(state,prev_node, next_node, weights, K_psi, K_phi, max_bank):
	# A: position of ac
	# P: prev_node 
	# Q: next_node
	# R: closest point on line to position

	# weights balance going to the line versus keeping heading
	# [0] : get to node. [1] get to line.

	def vec2heading(vec):
		angle = np.arctan2(vec[0],vec[1])
		return np.mod(angle + 2.*np.pi,2.*np.pi) 
	
	# points of interest
	A = np.array(state[:2])
	P = np.array(prev_node)
	Q = np.array(next_node)

	# command vector
	PQ = Q-P
	PA = A-P
	PR = PA.dot(PQ)/np.sum(PQ**2) * PQ 
	AR = PR - PA;
	
	# command psi
	psi_command = vec2heading(weights[0]*PQ + weights[1]*AR)
	
	# get state
	phi = state[3]
	psi = state[5]
	
	# error on psi
	err_psi = psi_command-psi
	
	# error on phi
	phi_command = np.maximum(-max_bank,np.minimum(max_bank, K_psi*err_psi))
	err_phi = phi_command-phi

	# control on phi
	u_phi = K_phi * err_phi
	return u_phi







