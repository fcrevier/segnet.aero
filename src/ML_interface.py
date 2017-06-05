#import googleImage
import urllib, StringIO
from math import log, exp, tan, atan, pi, ceil
from PIL import Image
import pdb
import numpy as np
import caffe
import deploy
import cv2

b_plotting = False


# Toby Buckley 
# interface between the state and the machine learning

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
    
def indexing_with_clipping(arr, indices, clipping_value=0):
    idx = np.where(indices < arr.shape,indices,clipping_value)
    return arr[idx[:, 0], idx[:, 1]]

def getSubMap(x, y, globalMap):
  #x, y are the plane's coordinates
  arr = np.array(globalMap)
  x0 = max(int(round(x - 375.0/2)), 0)
  x1 = max(int(round(x + 375.0/2)), 0)
  y0 = max(int(round(y - 375.0/2)), 0)
  y1 = max(int(round(y + 375.0/2)), 0)
  area = (x0, y0, x1, y1)
  dx = x1-x0
  dy = y1-y0

  #Map = np.zeros((375, 375))
  bidx = [x0, y0]
  eidx = [x1, y1]
  cropSliceArr = np.s_[bidx[0]:eidx[0],bidx[1]:eidx[1],:]
  #cropSliceMap = np.s_[]
  Map = arr[cropSliceArr]

  localMap = Image.fromarray(Map)
  return localMap


def getImage(x,y):
    scale = 1
    zoom = 18
    sizex = 375
    sizey = 375
    latn = x
    lonn = y
    position = ','.join((str(latn), str(lonn)))
    urlparams = urllib.urlencode({'center': position, #png is default
                                  'zoom': str(zoom),
                                  'size': '%dx%d' % (sizex, sizey),
                                  'maptype': 'satellite',
                                  'sensor': 'false',
                                  'scale': scale})
    url = 'http://maps.google.com/maps/api/staticmap?' + urlparams
    f=urllib.urlopen(url)
    im=Image.open(StringIO.StringIO(f.read()))
    return im

def initNet():
    deploy_file = '/home/tobyb/ML/segnet.aero/SegNet/DIGITS/best/deploy.prototxt'
    caffemodel = '/home/tobyb/ML/segnet.aero/SegNet/DIGITS/best/snapshot_iter.caffemodel'
    mean_file = '/home/tobyb/ML/segnet.aero/SegNet/DIGITS/best/mean.binaryproto'
    use_gpu = True
    net = deploy.get_net(caffemodel, deploy_file, use_gpu)
    transformer = deploy.get_transformer(deploy_file, mean_file)
    return net, transformer

def feedCNN(net, im_input):
    net.blobs['data'].data[...] = im_input
    out = net.forward()
    pred = out['conv_classifier']
    sq = np.squeeze(pred[0,:,:,:]) #2x375x375
    ind = np.argmax(sq, axis=0)
    
    outImg = Image.fromarray(255*ind.astype('uint8'))
    outImg.show()
    pdb.set_trace()
    return pred

def getPredInd(pred):
    sq = np.squeeze(pred[0,:,:,:]) #2x375x375
    ind = np.argmax(sq, axis=0)
    outImg = Image.fromarray(255*ind.astype('uint8'))
    outImg = outImg.convert('RGB')
    if b_plotting:
        outImg.show()
    return ind, outImg


def updateMap(state, net, transformer, globalMap):
    #pass in the neural net
    x,y = state
    #assume x, y are in latitude, longitude

    #get image from google
    img = getSubMap(x,y, globalMap) 

    #test image from our dataset
    #img = Image.open('/home/tobyb/ML/cs231nProject/SegNet/CamVid/test/10378780_15_01_04.png')
    img = img.convert('RGB')
    if b_plotting:
        img.show()
    #feed image into CNN
    im_input = np.array(img)
    #im_input = np.transpose(im_input, (2,0,1))
    #im_input = np.expand_dims(im_input, axis=0)
    scores = deploy.forward_pass([im_input], net, transformer)
    predInd, indImage = getPredInd(scores) #get prediction indices (array of 1's and 0's)
    #images start at left-upper corner, x=down, y=right, so rotate 90 degrees
    predInd = np.rot90(predInd, k=-1)

    return predInd, indImage, img

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

    return lines[i], minDist


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




