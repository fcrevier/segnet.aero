
from ML_interface import *
import pygame
from PIL import ImageOps
import pdb


x = 37.427466
y = -122.170288
x = 33.4315927
y = -86.7948165
#yellowstone:
x = 44.41092
y = -110.6718564
endx = 44.4422806
endy = -110.5677746



globalMapOriginal = Image.open("globalMap.png")
#pad with zeros, so that the airplane can fly near the edge
globalMap = ImageOps.expand(globalMapOriginal, border=188, fill='black')
globalW, globalH = globalMap.size

x0 = (globalW) / 2 #start in middle
y0 = (globalH) / 2 #start in middle

state = [x0,y0]

net, transformer = initNet()
predInd, indImage, img = updateMap(state, net, transformer, globalMap)
#get lines
lines = detectLines(indImage, indImage.mode)
pdb.set_trace()
