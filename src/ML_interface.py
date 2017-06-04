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

def detectLines(PILimage, imgType):
    #input is PIL image
    array = np.array(PILimage)
    if imgType == 'GRAY':
        pass
    elif imgType == 'BGR':
        img = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif imgType == 'RGB':
        img = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    #set up Hough transform
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    
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



def getDistError(state, predInd):
    x, y = state

    #feed output to linear regressor
    xind, yind = np.where(predInd)  # now +x is google map right, +y is google map up
    # ... need parameterized polyfit, so that vertical lines are valid
    m, y_intercept = 0,0 #np.polyfit(xind, yind, 1)
    # http://www.intmath.com/plane-analytic-geometry/perpendicular-distance-point-line.php
    dist = abs(y_intercept + m*375.0/2 + -1*375.0/2) / np.sqrt(m**2 + 1)
    
    # get the two points on either end of the section
    x0 = 0.0
    y0 = m*x0+y_intercept
    x1 = 375.0
    y1 = m*x1+y_intercept
    line = [x0, y0, x1, y1]
    # transform to global
    lineGlobal = line[:]
    for i in range(4):
        lineGlobal[i] -= 375.0/2
    lineGlobal[0] -= x
    lineGlobal[2] -= x
    lineGlobal[1] -= y
    lineGlobal[3] -= y

    return dist, line, lineGlobal




