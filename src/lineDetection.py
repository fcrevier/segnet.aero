# Line detection from segmentation labelling
from cnn_segmentation import *
import cv2
import numpy as np
import urllib, StringIO
from math import log, exp, tan, atan, pi, ceil
from PIL import Image

b_plotting = False

def splitHoughLines(lines, graph):
    #check each line against others for intersection, splits lines at intersection
    # graph = graph of nodes and edges
    # graph[nodeNum] = [connectedNodes, location]
    # this function should only be called once per tile
    graph = {}
    edges = {} #key = (node1, node2). output = edge distance
    numLines = lines.shape[0]
    bIntersect = np.zeros((numLines, numLines), dtype=bool)
    nodeCount = 1
    for i in range(numLines):
        #first endpoint
        node1 = nodeCount
        #second endpoint
        node2 = nodeCount + 1
        nodeCount += 2
        nodes = [] #nodes connected to
        intersections = getIntersections(lines)
        if len(intersections) == 0:
            graph[node1] = node2
            graph[node2] = node1


def combineHoughLines(lines):
    threshold = 30**2 #in pixels
    bKeep = np.ones(lines.shape[0], dtype=bool)
    for i in range(lines.shape[0]):
        if i==0: continue
        #compare to all previous lines
        for j in range(i):
            if bKeep[j] == False: #skip deleted lines
                continue
            d1 = (lines[i][0][0] - lines[j][0][0])**2 + \
                    (lines[i][0][1] - lines[j][0][1])**2
            d2 = (lines[i][0][2] - lines[j][0][2])**2 + \
                    (lines[i][0][3] - lines[j][0][3])**2
            if d1 < threshold and d2 < threshold:
                bKeep[j] = False
            #pdb.set_trace()
    return lines[bKeep]


def plotHoughLines(img, lines):
    a,b,c = lines.shape
    for i in range(a):
        cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow('result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectLines(PILimage, imgType):
    #input is PIL image
    array = np.array(PILimage)
    if imgType == 'GRAY':
        img = array
    elif imgType == 'BGR':
        img = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif imgType == 'RGB':
        img = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    #set up Hough transform
    minLineLength = 20
    maxLineGap = 5
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    array = cv2.resize(array, (0,0), fx=0.5, fy=0.5)
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 50, lines=np.array([]), \
        minLineLength=minLineLength, maxLineGap=maxLineGap)
    lines = combineHoughLines(lines) 
    numLines = lines.shape[0]
    if False:
        print(numLines)
        plotHoughLines(array, lines)
    return 2*lines #mult by 2 to compensate for 1/2 downsampling

def getGlobalLines(state, lines):
    # transform to global
    x, y = state[0:2]
    globalLines = np.copy(lines)
    for j in range(lines.shape[0]):
        for i in range(4):
            globalLines[j,0,i] -= 375.0/2
        globalLines[j,0,0] += y
        globalLines[j,0,2] += y
        globalLines[j,0,1] += x
        globalLines[j,0,3] += x
    #pdb.set_trace()
    return globalLines