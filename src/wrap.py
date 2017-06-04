
from ML_interface import *
import pygame


x = 37.427466
y = -122.170288
x = 33.4315927
y = -86.7948165
#yellowstone:
x = 44.41092
y = -110.6718564
endx = 44.4422806
endy = -110.5677746
state = x,y


net, transformer = initNet()
dist, predInd, indImage, img, line, lineGlobal = getDistError(state, net, transformer)
#get lines
#detectLines(indImage, 'GRAY')
img.show()

mode = img.mode
size = img.size
data = img.tobytes()
pygame.image.fromstring(data,size,mode)
