#archive
def getLocalImage(x,y):
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

def feedCNN(net, im_input):
    net.blobs['data'].data[...] = im_input
    out = net.forward()
    scores = out['conv_classifier']
    sq = np.squeeze(pred[0,:,:,:]) #2x375x375
    ind = np.argmax(sq, axis=0)
    return scores

def indexing_with_clipping(arr, indices, clipping_value=0):
    idx = np.where(indices < arr.shape,indices,clipping_value)
    return arr[idx[:, 0], idx[:, 1]]