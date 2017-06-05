

import stitchMap




#some of the code for tracking the lines on a global map

#global params
long0 = float(raw_input('lower-left longitude: \n'))
lat0 = float(raw_input('lower-left latitude: \n'))
global_width = 0.05 #degrees longitude
global_height = 0.02 #degrees latitude
long1 = long0 + global_width
lat1 = lat0 + global_height
zoom = 17

#get image of map
globalMap = stitchMap.getMap(long0, lat0, global_width, global_height, zoom)

#must put the following in the pygame initialize section
globalPoints = []
net, transformer = initNet()

def drawFrame():
    
    #fill black
    screen.fill(black)

    #draw global map - includes map lines
    screen.blit(globalMap, globalMapRect)
    drawMapLines(surface, globalPoints)

    #draw local map - includes map lines
    screen.blit(localMap, localMapRect)
    drawMapLines(surface, [line])

    #add plane to both global and local maps

    

    #draw ML output


def drawMapLines(surface, lines):
    for line in lines:
        x0 = (line[0] - long0) / (long1 - long0) * window_w
        x1 = (line[2] - long0) / (long1 - long0) * window_w
        #transform y because pygame starts from top-left, not bottom-left
        y0 = window_h - (line[1] - lat0) / (lat1 - lat0) * window_h
        y1 = window_h - (line[3] - lat0) / (lat1 - lat0) * window_h
        start_pos = [x0, y0]
        end_pos = [x1, y1]
        pygame.draw.line(surface, (255,255,255), start_pos, end_pos, width=5)


#must put the rest of this inside the pygame loop
state = x,y #current x,y lat. long. position
dist, predInd, line, lineGlobal = getDistError(state, net, transformer)
globalPoints.append(lineGlobal)


#at end of pygame main loop
drawFrame()
