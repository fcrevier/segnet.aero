# Basic OBJ file viewer. needs objloader from:
#  http://www.pygame.org/wiki/OBJFileLoader
# LMB + move: rotate
# RMB + move: pan
# Scroll wheel: zoom in/out
import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import matplotlib.pyplot as plt
import flight_dynamics as fd
# IMPORT OBJECT LOADER
from obj_loader import *

# CONSTANTS
D2R = np.pi/180.

pygame.init()
viewport = (800,600)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)           # most obj files expect to be smooth-shaded

# LOAD OBJECT AFTER PYGAME INIT
obj = OBJ('C:/Users/Felix/Documents/Dev/CS231n/segnet.aero/Env/uav_small.obj', swapyz=True)

clock = pygame.time.Clock()

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width/float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rx, ry = (0,0)
tx, ty = (0,0)
zpos = 10
rotate = move = False

FPS = 20
dt = 1./float(FPS)
dt = 0.1
# Initial conditions 
npts = 100
state = fd.initial_state()
state_history = np.zeros((npts,len(state)))
t = np.linspace(0., dt*npts, num = npts)
# objective
prev_node = (0.,0.)
next_node = (100., 100.)
for ii in range(npts):
    clock.tick(FPS)
    for e in pygame.event.get():
        if e.type == QUIT:
            sys.exit()
        elif e.type == KEYDOWN and e.key == K_ESCAPE:
            sys.exit()
        elif e.type == MOUSEBUTTONDOWN:
            if e.button == 4: zpos = max(1, zpos-1)
            elif e.button == 5: zpos += 1
            elif e.button == 1: rotate = True
            elif e.button == 3: move = True
        elif e.type == MOUSEBUTTONUP:
            if e.button == 1: rotate = False
            elif e.button == 3: move = False
        elif e.type == MOUSEMOTION:
            i, j = e.rel
            if rotate:
                rx += i
                ry += j
            if move:
                tx += i
                ty -= j


    # Clear and load object            
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # get state
    X, Y, Z, phi, theta, psi = state[:6]    

    # keep psi in [0 2pi)
    if psi > 2.*np.pi:
        psi = np.mod(psi,2.*np.pi)

    # compute body x vector (no theta)
    xb = (np.cos(psi), np.sin(psi), 0.)


    glTranslate(X, Y, - zpos)
    glRotate(phi/D2R,*xb)
    glRotate(psi/D2R,0,0,1)
    glTranslate(0., 0., 0.)
    

    # TURN MANEUVER
    # controller
    K_psi = -1
    K_phi = 1
    max_bank = 30.*D2R
    weights = (2.,1.)
    u_phi = fd.follow_line(state,
                        prev_node, next_node, weights, 
                        K_psi, K_phi, max_bank)
    
    # plant dynamics
    Ixx = 1.
    state_dot = fd.turn(state, u_phi, Ixx)

    # update
    state = fd.next_state(state, state_dot, dt)
    state_history[ii,:] = np.array(state)

    # SHOW OBJECT
    glCallList(obj.gl_list)

    # FLIP DISPLAY
    pygame.display.flip()

# PLOT HISTORY
plt.plot(t,state_history[:,5])
plt.show()