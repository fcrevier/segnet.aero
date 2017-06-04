
from ML_interface import *
import pygame
import stitchMap

x = 37.427466
y = -122.170288
x = 33.4315927
y = -86.7948165
#yellowstone:
lat0 = 44.41092
long0 = -110.6718564
global_height = 44.4422806 - lat0
global_width = -110.5677746 - long0
zoom = 17

globalMap = stitchMap.getMap(long0, lat0, global_width, global_height, zoom)
globalMap.save('globalMap.png')