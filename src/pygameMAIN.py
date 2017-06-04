import os,sys,copy,random,math,time
import pygame
from pygame.locals import *
from PIL import Image, ImageOps
import stitchMap, GNC, ML_interface
import numpy as np
import pdb
os.environ['GLOG_minloglevel'] = '3'

def PIL2surf(img):
    mode = img.mode
    size = img.size
    data = img.tobytes()
    surf = pygame.image.fromstring(data, size, mode) #creates surface from string
    #pdb.set_trace()
    return surf


def drawFrame(screen, lines, surf_localMap, surf_outputMap):
    
    #fill black
    screen.fill((0,0,0))

    #draw global map - includes map lines
    w, h = globalMapRect[2:4]
    #drawMapLines(surf_globalMap, w, h, globalPoints)
    screen.blit(surf_globalMap, globalMapRect)

    #draw local map - includes map lines
    w, h = localMapRect[2:4]
    pic = pygame.transform.scale(surf_localMap, (w,h))
    #drawMapLines(surface, lines)
    screen.blit(pic, localMapRect)

    #draw local map - includes map lines
    w, h = outputMapRect[2:4]
    pic = pygame.transform.scale(surf_outputMap, (w,h))
    screen.blit(pic, outputMapRect)

    #add plane to both global and local maps

    

    #draw ML output


def drawMapLines(surface, window_w, window_h, lines):
    for line in lines:
        x0 = (line[0] - long0) / (long1 - long0) * window_w
        x1 = (line[2] - long0) / (long1 - long0) * window_w
        #transform y because pygame starts from top-left, not bottom-left
        y0 = window_h - (line[1] - lat0) / (lat1 - lat0) * window_h
        y1 = window_h - (line[3] - lat0) / (lat1 - lat0) * window_h
        start_pos = [x0, y0]
        end_pos = [x1, y1]
        pygame.draw.line(surface, (255,255,255), start_pos, end_pos, width=5)



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
long0 = -122 #float(raw_input('lower-left longitude: \n'))
lat0 =  37.427 #float(raw_input('lower-left latitude: \n'))
#global_width = 0.01 #degrees longitude
#global_height = 0.02 #degrees latitude
#long1 = long0 + global_width
#lat1 = lat0 + global_height
zoom = 17

#get image of map
#globalMap = stitchMap.getMap(long0, lat0, global_width, global_height, zoom)
globalMap = Image.open("globalMap.png")
#pad with zeros, so that the airplane can fly near the edge
globalMap = ImageOps.expand(globalMap, border=188, fill='black')
globalW, globalH = globalMap.size
surf_globalMap = PIL2surf(globalMap)


#pygame globals
temp_global=0
scrsize=[1500,1000] #in pixels
aspratio=scrsize[1]/scrsize[0]
clock=pygame.time.Clock()
FPS=30
FPS_map = 30
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
#resize global map
surf_globalMap = pygame.transform.scale(surf_globalMap, globalMapRect[2:4])


#init
os.environ['SDL_VIDEO_WINDOW_POS'] = '50,50'
pygame.init()
#logo=pygame.image.load("logo filename")
#pygame.display.set_icon(logo)
pygame.display.set_caption('Auto UAV')
screen=pygame.display.set_mode(scrsize) #base surface
ipics = 0

#machine learning INIT
globalPoints = []
net, transformer = ML_interface.initNet()
x0 = (globalW) / 2 #start in middle
y0 = (globalH) / 2 #start in middle
print(x0,y0)
state = [x0, y0, 50]  #initial x,y lat. long. position, z=altitude in km
action = 0
offset = 0,0 #idk, must update, function of zoom
mapTimer = 9999
dist = 0.0
#main loop:
while 1:
    seconds=clock.tick(FPS)/1000.0 #how many seconds passed since previous call

    #update the state
    state = GNC.getNextState( state, action)
    #check state --- constrain to inner box
    x, y = state[0:2]
    x, y = checkXY(x,y)




    predInd, indImage, localImage = \
            ML_interface.updateMap((x, y), net, transformer, globalMap)
    if mapTimer < 1.0 / FPS_map:
        mapTimer += seconds
    else:
        dist, line, lineGlobal = ML_interface.getDistError((x,y),predInd)
        surf_outputMap = PIL2surf(indImage)
        surf_localMap = PIL2surf(localImage)
        globalPoints.append(lineGlobal)
        print('Updated Map')
        mapTimer = 0.0
    #get next action from a controller
    action = GNC.getNextAction(state, dist)

    #at end of pygame main loop
    drawFrame(screen, [line], surf_localMap, surf_outputMap)

    #pygame.image.save(screen, 'pics\image'+str(ipics)+'.png')
    ipics += 1
    pygame.display.flip() #show map