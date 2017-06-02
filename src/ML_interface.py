#import googleImage
import urllib, StringIO
from math import log, exp, tan, atan, pi, ceil
from PIL import Image
import pdb
import numpy as np
import caffe
import deploy

b_plotting = False


# Toby Buckley 
# interface between the state and the machine learning





def getImage(x,y):
    scale = 1
    zoom = 17
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
    mean_file = '/home/tobyb/ML/segnet.aero/SegNet/DIGITS/mean.binaryproto'
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
    if b_plotting:
        outImg.show()
    return ind


def getDistError(state, net, transformer):
    #pass in the neural net
    x,y = state
    #assume x, y are in latitude, longitude

    #get image from google
    img = getImage(x,y)

    #test image from our dataset
    img = Image.open('/home/tobyb/ML/cs231nProject/SegNet/CamVid/test/10378780_15_01_04.png')
    img = img.convert('RGB')
    if b_plotting:
        img.show()
    #feed image into CNN
    im_input = np.array(img)
    #im_input = np.transpose(im_input, (2,0,1))
    #im_input = np.expand_dims(im_input, axis=0)
    scores = deploy.forward_pass([im_input], net, transformer)
    predInd = getPredInd(scores) #get prediction indices (array of 1's and 0's)
    #images start at left-upper corner, x=down, y=right, so rotate 90 degrees
    predInd = np.rot90(predInd, k=-1)
    #feed output to linear regressor
    xind, yind = np.where(predInd)  # now +x is google map right, +y is google map up
    # ... need parameterized polyfit, so that vertical lines are valid
    m, y_intercept = np.polyfit(xind, yind, 1)
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

    return dist, predInd, line, lineGlobal








