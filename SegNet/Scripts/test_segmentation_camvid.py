import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
import time
import pdb
from sklearn.preprocessing import normalize
caffe_root = '/home/tobyb/ML/caffe-segnet-cudnn5/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

error = 0
notroad = 0
roadaccuracy = 0
count = args.iter

for i in range(0, args.iter): #how many images to do

	t = time.time()
	net.forward()
	dt = time.time() - t
	print('Forward pass time: %.2f ms' % (1000.0*dt))

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data #softmax prediction values 1x2x375x375
	notroad += np.sum(label) / (375*375)
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0) #1x375x375 index classes
	label_squeezed = np.squeeze(label[0,:,:,:])
	error += np.sum(abs(label_squeezed-ind)/(375*375))
	if np.sum(label) > 0: #only proceeds if there's any road on true label
		roadaccuracy += np.sum(ind[label_squeezed.astype(bool)])/(np.sum(label_squeezed))
	else:
		count -= 1
	#pdb.set_trace()

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()

	Not_Road = [128,128,128]
	Road = [128,0,0]

	label_colours = np.array([Not_Road, Road])
	for l in range(0,2):
		r[ind==l] = label_colours[l,0]
		g[ind==l] = label_colours[l,1]
		b[ind==l] = label_colours[l,2]
		r_gt[label==l] = label_colours[l,0]
		g_gt[label==l] = label_colours[l,1]
		b_gt[label==l] = label_colours[l,2]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = r/255.0
	rgb[:,:,1] = g/255.0
	rgb[:,:,2] = b/255.0
	rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb_gt[:,:,0] = r_gt/255.0
	rgb_gt[:,:,1] = g_gt/255.0
	rgb_gt[:,:,2] = b_gt/255.0

	image = image/255.0

	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]


	#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save(IMAGE_FILE+'_segnet.png')

	if True:
		plt.figure()
		plt.imshow(image,vmin=0, vmax=1)
		plt.figure()
		plt.imshow(rgb_gt,vmin=0, vmax=1)
		plt.figure()
		plt.imshow(rgb,vmin=0, vmax=1)
		plt.show()


print 'Success!'

print('Avg acc: %.2f' % (1-error/args.iter))
print('Avg truth not road percent: %.2f' % (1- notroad/args.iter))
print('Avg road accuracy: %.2f' % (roadaccuracy/count))

