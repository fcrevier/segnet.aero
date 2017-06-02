
from ML_interface import *



x = 37.427466
y = -122.170288
state = x,y

net, transformer = initNet()
getDistError(state, net, transformer)
