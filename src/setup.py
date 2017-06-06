
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
zoom = 18


# massachusetts
lat0 = 41.9490165
long0 = -71.3269226
lat1 = 41.9910937
long1 = -71.2671912
global_height = lat1 - lat0
global_width = long1 - long0

# highway 120 to yosemite
lat0 = 37.8117537
long0 = -119.9589972
lat1 = 37.8248696
long1 = -119.912249




global_height = lat1 - lat0
global_width = long1 - long0

globalMap = stitchMap.getMap(long0, lat0, global_width, global_height, zoom)
globalMap.save('globalMap.png')
