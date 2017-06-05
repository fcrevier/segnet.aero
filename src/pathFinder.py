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
