import numpy as np

def pDistance(x, y, x1, y1, x2, y2):
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    A = x-x1
    B = y-y1
    C = x2-x1
    D = y2-y1
    dot = A*C+B*D
    len_sq = C*C+D*D
    param = -1
    if len_sq != 0:
        param = dot / len_sq
    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param*C
        yy = y1 + param*D
    dx = x-xx
    dy = y-yy
    return np.sqrt(dx*dx+dy*dy)


def getDistError(state, lines):
    if lines.shape[0] == 0:
        return lines, 0
    x, y = state

    numLines = lines.shape[0]
    minDist = 9999999
    idx = 0
    for i in range(numLines):
        line = lines[i]
        dist = pDistance(x, y, line[0,0], line[0,1], line[0,2], line[0,3])
        if dist < minDist:
            minDist = dist
            idx = i

    #only get first line
    line = np.expand_dims(lines[i], axis=0)
    return line, minDist

def affine(t1, t2, alpha):
    if len(t1) != len(t2): 
        print 'error, tuples not of same length'
    return tuple(t1[i]+alpha*t2[i] for i in range(len(t1)))