# CNN segmentation classifier
import numpy as np
import caffe
import deploy
import urllib, StringIO
from PIL import Image

b_plotting = False

def initNet():
    deploy_file = '../SegNet/DIGITS/best/deploy.prototxt'
    caffemodel = '../SegNet/DIGITS/best/snapshot_iter.caffemodel'
    mean_file = '../SegNet/DIGITS/best/mean.binaryproto'
    use_gpu = False
    net = deploy.get_net(caffemodel, deploy_file, use_gpu)
    transformer = deploy.get_transformer(deploy_file, mean_file)
    return net, transformer

def getSegmented(scores):
    sq = np.squeeze(scores[0,:,:,:]) #2x375x375
    pred_class = np.argmax(sq, axis=0)
    return pred_class

def segImage(img_rgb, net, transformer):
    
    #test image from our dataset
    #img = Image.open('/home/tobyb/ML/cs231nProject/SegNet/CamVid/test/10378780_15_01_04.png')
    if b_plotting:
        img_rgb.show()

    # to numpy
    img_inp = np.array(img_rgb)
    
    # feed through cnn
    scores = deploy.forward_pass([img_inp], net, transformer)
    pred_class = getSegmented(scores) #get prediction class (o or 1)
    
    # from class indices to image
    img_seg = Image.fromarray(255*pred_class.astype('uint8')).convert('RGB')

    #images start at left-upper corner, x=down, y=right, so rotate 90 degrees
    ## rotated index but not image  ?????
    pred_class = np.rot90(pred_class, k=-1)

    return img_seg