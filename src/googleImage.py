
import urllib, StringIO
from math import log, exp, tan, atan, pi, ceil
from PIL import Image
import pdb


scale = 1
zoom = 16

sizex = 375
sizey = 375

latn = 37.427466
lonn = -122.170288
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
im.show()
