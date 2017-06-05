# IMPORT GENERAL PACKAGES
import os,sys,copy,random,math,time
import pygame
from pygame.locals import *
from PIL import Image, ImageOps
import numpy as np
import pdb

# IMPORT CUSTOM PACKAGES
import stitchMap, imageTools, segmentation, lineDetection, GNC

# USEFUL FUNCTIONS
def PIL2surf(img):
    mode = img.mode
    size = img.size
    data = img.tobytes()
    surf = pygame.image.fromstring(data, size, mode) #creates surface from string
    #pdb.set_trace()
    return surf

def drawFrame(screen, lines, surf_loc, surf_seg):
    
    #fill black
    screen.fill((0,0,0))

    #draw global map - includes map lines
    w, h = globalMapRect[2:4]
    drawMapLines(surf_globalMap, globalPoints)
    picGlobal = pygame.Surface((globalOW, globalOH))
    picGlobal.blit(surf_globalMap, (0,0), (188,188,globalOW,globalOH))

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


#pygame globals
temp_global=0
scrsize=[1500,1000] #in pixels
aspratio=scrsize[1]/scrsize[0]
clock=pygame.time.Clock()
FPS=30
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
ipics = 0

#machine learning INIT
globalPoints = []
net, transformer = segmentation.initNet()
x0 = (globalW) / 2 #start in middle
y0 = (globalH) / 2 #start in middle
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


# GAME FEATURE SELECTOR
find_lines = False
seg_image = False

# MAIN LOOP
done = False
while 1:
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


    seconds=clock.tick(FPS)/1000.0 #how many seconds passed since previous call

    #update the state
    state = GNC.getNextState( state, action)
    #check state --- constrain to inner box
    x, y = state[0:2]
    x, y = checkXY(x,y)

    
    if mapTimer < 1.0 / FPS_map:
        mapTimer += seconds
    else:       
        # local image
        img_local = imageTools.getLocalImage(x, y, globalMap, (375., 375.))
        surf_loc = PIL2surf(img_local)
        
        # segmented image
        if seg_image :
            img_seg = segmentation.segImage(img_local, net, transformer)
            surf_seg = PIL2surf(img_seg)
        else:
            surf_seg = None

        # Find lines
        if find_lines:
            lines = lineDetection.detectLines(img_seg, img_seg.mode)
            lines = lineDetection.splitLines(lines)
            dist = lineDetection.getDistError((x,y),lines)
            globalLines = lineDetection.getGlobalLines(state, lines)
        else:
            lines = None

        print('Updated Map')

        mapTimer = 0.0

    if globalDrawTimer < 1.0 / FPS_drawGlobal:
        globalDrawTimer += seconds
   
    else:
        globalDrawTimer = 0.0

    #get next action from a controller
    action = GNC.getNextAction(state, 0.)

    #at end of pygame main loop
    drawFrame(screen, lines = lines, surf_loc = surf_loc, surf_seg = surf_seg)

    #pygame.image.save(screen, 'pics\image'+str(ipics)+'.png')
    ipics += 1
    pygame.display.flip() #show map
