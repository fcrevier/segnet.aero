
from ML_interface import *
import pygame
from PIL import ImageOps
import pdb


x = 37.427466
y = -122.170288
x = 33.4315927
y = -86.7948165
#yellowstone:
x = 44.41092
y = -110.6718564
endx = 44.4422806
endy = -110.5677746


def getSegmented(scores):
    sq = np.squeeze(scores[0,:,:,:]) #2x375x375
    pred_class = np.argmax(sq, axis=0)
    return pred_class

img_rgb = Image.open("Selection_011.png").convert('RGB')

net, transformer = initNet()
# to numpy
img_inp = np.array(img_rgb)
    
    # feed through cnn
scores = deploy.forward_pass([img_inp], net, transformer)
pred_class = getSegmented(scores) #get prediction class (o or 1)
    
    # from class indices to image
img_seg = Image.fromarray(255*pred_class.astype('uint8')).convert('RGB')
img_seg.show()
