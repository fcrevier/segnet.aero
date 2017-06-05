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


# IMPORT OBJECT LOADER
from objFileLoader import *

# CONSTANTS
D2R = np.pi/180.

# INIT  GAME
pygame.init()
clock = pygame.time.Clock()
viewport = (800,600)
hx = viewport[0]/2
hy = viewport[1]/2
srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
zpos = 10
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

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
width, height = viewport
gluPerspective(90.0, width/float(height), 1, 100.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_MODELVIEW)

rect = Rect((10, 10), (100, 100))

FPS = 20
dt = 1./float(FPS)
X, Y, phi, psi = (0.,0., 30., 45.) 
while True:
    # Clear and load object            
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # get state
    phi -= 1
    # keep psi in [0 2pi)
    #if psi > 2.*np.pi:
     #   psi = np.mod(psi,2.*np.pi)

    # compute body x vector (no theta)
    xb = (np.cos(psi), np.sin(psi), 0.)
    
    glTranslate(X, Y, - zpos)
    glRotate(phi,*xb)
    glRotate(psi,0,0,1)
    glTranslate(0., 0., 0.)

    # SHOW OBJECT
    glCallList(obj.gl_list)
    srf_current = pygame.display.get_surface()
    pygame.draw.rect(srf_current,(200,200,200),rect)
    # FLIP DISPLAY
    pygame.display.flip()