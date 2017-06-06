# IMPORT GENERAL PACKAGES
import os,sys,copy,random,math,time
import pygame
from pygame.locals import *
from PIL import Image, ImageOps
import numpy as np
import pdb
import util
import matplotlib.pyplot as plt

# IMPORT CUSTOM PACKAGES
import stitchMap, imageTools, segmentation, lineDetection
import flight

# CONSTANTS
D2R = np.pi/180.
PIX2M = 2.1

# USEFUL FUNCTIONS
def PIL2surf(img):
    mode = img.mode
    size = img.size
    data = img.tobytes()
    surf = pygame.image.fromstring(data, size, mode) #creates surface from string
    #pdb.set_trace()
    return surf

def drawFrame(state, screen, lines, surf_loc, surf_seg):
    
    x,y = state[0:2]
    #fill black
    screen.fill((0,0,0))

    #draw global map - includes map lines
    w, h = globalMapRect[2:4]
    drawMapLines(surf_globalMap, globalPoints)
    pygame.draw.circle(surf_globalMap, (0,0,255), (x,y), 10)
    if bDrawGlobalMap:
        picGlobal = pygame.Surface((globalOW, globalOH))
        picGlobal.blit(surf_globalMap, (0,0), (188,188,globalOW,globalOH))
    else:
        picGlobal = pygame.Surface((375*3, 375*3))
        x0 = x - int(375*3/2)
        y0 = y - int(375*3/2)
        picGlobal.blit(surf_globalMap, (0,0), (x0,y0,375*3, 375*3))

    #resize global map
    picGlobal = pygame.transform.scale(picGlobal, globalMapRect[2:4])
    screen.blit(picGlobal, globalMapRect)

    #draw local map - includes map lines
    w, h = localMapRect[2:4]
    if lines != None:
        drawMapLines(surf_loc, [lines])
    pic = pygame.transform.scale(surf_loc, (w,h))
    screen.blit(pic, localMapRect)

    #draw local map - includes map lines
    w, h = outputMapRect[2:4]
    if surf_seg != None:
        pic = pygame.transform.scale(surf_seg, (w,h))
        screen.blit(pic, outputMapRect)

    #add plane to both global and local maps

    #add text above each graph
    label = myfont.render("Global Map & Ground Track", 8, (255, 0, 0))
    screen.blit(label, (globalMapRect[2]/2-90, 10))

    label = myfont.render("CNN Input & Hough Lines", 8, (255, 0, 0))
    screen.blit(label, (globalMapRect[2]-25+localMapRect[2]/2, 10))

    label = myfont.render("Segmentation Output", 8, (255, 0, 0))
    screen.blit(label, (globalMapRect[2]-25+localMapRect[2]/2, 15+marginAmount[1]+localMapRect[3]))


def drawMapLines(surface, linesGroup):
    for lines in linesGroup:
        for i in range(lines.shape[0]):
            x0 = lines[i,0,0]
            y0 = lines[i,0,1]
            x1 = lines[i,0,2]
            y1 = lines[i,0,3]
            start_pos = [x0, y0]
            end_pos = [x1, y1]
            pygame.draw.line(surface, (0,0,255), start_pos, end_pos, 5)



def checkXY(x, y):
  if x < 188:
    x = 188
  elif x > globalW - 187:
    x = globalW - 187

  if y < 188:
    y = 188
  elif y > globalH - 187:
    y = globalH - 187
  return x,y





#global params
long0 = -122 
lat0 =  37.427
zoom = 17

#get image of map
#globalMap = stitchMap.getMap(long0, lat0, global_width, global_height, zoom)
globalMapOriginal = Image.open("globalMap.png")
globalOW, globalOH = globalMapOriginal.size
#pad with zeros, so that the airplane can fly near the edge
globalMap = ImageOps.expand(globalMapOriginal, border=188, fill='black')
globalW, globalH = globalMap.size
surf_globalMap = PIL2surf(globalMap)
bDrawGlobalMap = False


#pygame globals
temp_global=0
scrsize=[1500,1000] #in pixels
aspratio=scrsize[1]/scrsize[0]
clock=pygame.time.Clock()
FPS=60
FPS_map = 5
FPS_drawGlobal = 0.25
seconds=0.0
milli=0.0
# params for the windows
marginRatio = 0.02 #percent
bigMapRatio = 2.0/3
inner = np.array(scrsize)
inner[0] *= bigMapRatio
marginAmount = np.array(scrsize) * marginRatio
globalMapRect = pygame.Rect( marginAmount, inner-2*marginAmount)
# CNN input
left = inner[0] + marginAmount[0]
upper = marginAmount[1]
width = (np.array(scrsize)[0] - inner[0]) - 2*marginAmount[0]
height = np.array(scrsize)[1] / 2.0 - 2*marginAmount[1]
localMapRect = pygame.Rect(left, upper, width, height)
# CNN output
upper = np.array(scrsize)[1] / 2.0 + marginAmount[1]
outputMapRect = pygame.Rect(left, upper, width, height)



#init
os.environ['GLOG_minloglevel'] = '3'
os.environ['SDL_VIDEO_WINDOW_POS'] = '50,50'
pygame.init()
#logo=pygame.image.load("logo filename")
#pygame.display.set_icon(logo)
pygame.display.set_caption('Auto UAV')
screen=pygame.display.set_mode(scrsize) #base surface
myfont = pygame.font.SysFont("monospace",20)
ipics = 0

#machine learning INIT
globalPoints = []
net, transformer = segmentation.initNet()
x0 = 1000 #2000 #1000.0 + 4000.0 #start in middle
y0 = (globalH) / 2.0 + 550.0 #start in middle
state = [x0, y0, 50]  #initial x,y lat. long. position, z=altitude in km
action = 0
offset = 0,0 #idk, must update, function of zoom
mapTimer = 9999
globalDrawTimer = 9999
dist = 0.0

#set up the path finding graph
class Tile:
  def __init__(self, x,y,w,h):
    pixelX = x
    pixelY = y
    pixelW = w
    pixelH = h
    nodes = []


tileDict = {} #dictionary where key = (tilex, tiley) 
numX = 20
numY = 20
#tileMap = initTileMap(globalMap, numX, numY) #tileMap[tilex, tiley] = tile class object

# AIRCRAFT PARAMETERS
aircraft = {'speed': 80.,
            'K_psi': -0.5,
            'K_phi' : 2.0, 
            'max_bank': 450.*D2R,
            'line_weights': (1.,3.) }

# INIT CONDITIONS 
X0, Y0, phi0, psi0 = (x0*PIX2M, y0*PIX2M, 0., 0.) 
state = flight.initial_state(*(X0, Y0, phi0, psi0))
state_history = np.zeros((2000,len(state)+1))
ii_plane = 0
prev_node = (X0, Y0)
next_node = (X0+10,Y0)

# GAME FEATURE SELECTOR
find_lines = True
seg_image = True

# MAIN LOOP
done = False
while 1:
    # TIME
    seconds=clock.tick(FPS)/1000.0 #how many seconds passed since previous call
    
    # GET EVENTS
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                done = True
                break 
        
        elif event.type == pygame.QUIT:
            done = True
            break 
    if done:
        break 


    

    #check state --- constrain to inner box
    x, y = int(state[0] / PIX2M), int(state[1] / PIX2M)
    x, y = checkXY(x,y)
    
    if mapTimer < 1.0 / FPS_map:
        mapTimer += seconds
    else:    
        mapTimer = 0.0

        # GET LOCAL IMAGE
        img_local = imageTools.getLocalImage(x, y, globalMap, (375., 375.))
        surf_loc = PIL2surf(img_local)
        
        # GET SEGMENTED IMAGE
        if seg_image :
            img_seg = segmentation.segImage(img_local, net, transformer)
            surf_seg = PIL2surf(img_seg)
        else:
            surf_seg = None

        # FIND LINES
        if find_lines:
            lines = lineDetection.detectLines(img_seg, img_seg.mode)
            #lines = lineDetection.splitLines(lines)
            lines, dist = util.getDistError((x,y),lines)
            #convert to meters
            linesDraw = lines
            #pdb.set_trace()
            lines = (lines - 375.0/2.0)
        else:
            lines = None

        print('Updated Map')        

        # GET STATE (CS origin : top left of global, X Right, Y Down)
        X, Y, phi, psi = state

        # OBJECTIVE (CS origin : center of local image)
        if lines.shape[0] != 0:
            bKeepHeading = False
            #pdb.set_trace()
            x0 = (lines[0][0,0]) * PIX2M
            y0 = (lines[0][0,1]) * PIX2M
            x1 = (lines[0][0,2]) * PIX2M
            y1 = (lines[0][0,3]) * PIX2M
            #pdb.set_trace()
            prev_node = (X+x0, Y+y0)
            next_node = (X+x1, Y+y1)
            #prev_node = (X, Y-10)
            #next_node = (X+10, Y-20)
        else:
            bKeepHeading = True


        

        # CONTROLLER

        if bKeepHeading:
            psi_c = psi
        else:
            psi_c = flight.followLine((X,Y), prev_node, next_node, aircraft['line_weights'])

        # PLANT DYNAMICS
        state_dot = flight.plant(state, aircraft, psi_c)

        # EULER UPDATE
        state = flight.next_state(state, state_dot, 1.0/FPS_map)
        state_history[ii_plane,:4] = np.array(state)
        state_history[ii_plane,4] = psi_c
        ii_plane += 1



    if globalDrawTimer < 1.0 / FPS_drawGlobal:
        globalDrawTimer += seconds
   
    else:
        if find_lines:
            globalLines = lineDetection.getGlobalLines(state, lines)
        globalDrawTimer = 0.0


    #at end of pygame main loop
    drawFrame((x, y), screen, lines = linesDraw, surf_loc = surf_loc, surf_seg = surf_seg)
    

    #pygame.image.save(screen, 'pics\image'+str(ipics)+'.png')
    ipics += 1
    pygame.display.flip() #show map
    #pdb.set_trace()

# PLOT HISTORY
# time vector
t = np.linspace(0., 1.0/FPS_map*(ii_plane+1), num = ii_plane) 

# objective line
#m = (next_node[1]-prev_node[1])/(next_node[0]-prev_node[0])
#b = next_node[1] - m * next_node[0]

# clip array to non-zero values
state_history = state_history[:(ii_plane),:]

# ground track
plt.plot(state_history[:,0],state_history[:,1],'k')#,state_history[:,0],b + m*state_history[:,0])
plt.show()

# attitude
plt.plot(t,state_history[:,[2,3,4]]/D2R)
plt.show()
