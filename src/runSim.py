# GENERAL IMPORTS
import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import matplotlib.pyplot as plt
import pdb

# IMPORT OBJECT LOADER
from objFileLoader import *

# IMPORT FLIGHT DYNAMICS
import flightDynamics as fd

# CONSTANTS
D2R = np.pi/180.
PIX2M = 2.1 # @17 zoom

# FRAME RATE 
FPS = 10
dt = 1./float(FPS)
npts = 1000
   

# INIT  GAME
pygame.init()
clock = pygame.time.Clock()
viewport = (1500,1000)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
zpos = 50

# INIT GRAPHICS
glLightfv(GL_LIGHT0, GL_POSITION,  (-40, 200, 100, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
glEnable(GL_LIGHT0)
glEnable(GL_LIGHTING)
glEnable(GL_COLOR_MATERIAL)
glEnable(GL_DEPTH_TEST)
glShadeModel(GL_SMOOTH)   # most obj files expect to be smooth-shaded

# LOAD OBJECT AFTER PYGAME INIT
obj = OBJ('uav_small.obj', swapyz=True)

# CHANGE PERSPECTIVE
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width/float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

# AIRCRAFT PARAMETERS
aircraft = {'speed': 7.,
            'K_psi': -0.2,
            'K_phi' : 2.0, 
            'max_bank': 30.*D2R,
            'line_weights': (1.5,1.) }

# INIT CONDITIONS 
X0, Y0, phi0, psi0 = (0.,0., 0., 0.) 
state = fd.initial_state(*(X0, Y0, phi0, psi0))
state_history = np.zeros((npts,len(state)+1))

ii = 0
done = False    
# GAME LOOP
while True:
    # GET EVENTS
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True
                break # break out of the for loop
        
        elif event.type == pygame.QUIT:
            done = True
            break # break out of the for loop
    if done:
        break # to break out of the while loop


    # clear and load graphics            
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # OBJECTIVE (get from map/ML)
    prev_node = (-10.,10.)
    next_node = (20., 30.)

    # get state
    X, Y, phi, psi = state

    # control
    if ii<20:
        psi_c = psi
    else:
        psi_c = fd.followLine((X,Y), prev_node, next_node, aircraft['line_weights'])

    state_dot = fd.plant(state, aircraft, psi_c)

    # update
    state = fd.next_state(state, state_dot, dt)
    state_history[ii,:4] = np.array(state)
    state_history[ii,4] = psi_c

    # APPLY GRAPHICS TRANSFORMS    
    # compute body x vector (no theta)
    xb = (np.cos(psi), np.sin(psi), 0.)
    glTranslate(X/PIX2M - 15, Y/PIX2M-10, - zpos)
    glRotate(phi/D2R,*xb)
    glRotate(psi/D2R,0,0,1)
    glTranslate(0., 0., 0.)

    # show
    glCallList(obj.gl_list)
    
    # FLIP DISPLAY
    pygame.display.flip()
    ii +=1
    #pdb.set_trace()

# PLOT HISTORY
t = np.linspace(0., dt*(ii+1), num = ii) 
# position
m = (next_node[1]-prev_node[1])/(next_node[0]-prev_node[0])
b = next_node[1] - m * next_node[0]

state_history = state_history[:(ii),:]
plt.plot(state_history[:,0],state_history[:,1],'k',state_history[:,0],b + m*state_history[:,0])
plt.show()

# attitude
plt.plot(t,state_history[:,[2,3,4]]/D2R)
plt.show()