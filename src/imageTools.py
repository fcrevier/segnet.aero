from PIL import Image
import numpy as np
import pdb


b_plotting = False


# Toby Buckley 
# interface between the state and the machine learning


def getLocalImage(x, y, img_global, loc_size = (375., 375.)):
    #x, y are the plane's coordinates
    arr_glob = np.array(img_global)
    x0 = max(int(round(x - loc_size[0]/2)), 0)
    x1 = max(int(round(x + loc_size[0]/2)), 0)
    y0 = max(int(round(y - loc_size[1]/2)), 0)
    y1 = max(int(round(y + loc_size[1]/2)), 0)
    area = (x0, y0, x1, y1)
    arr_loc = arr_glob[ y0:y1 , x0:x1 ,:]
    map_loc = Image.fromarray(arr_loc)
    img_loc = map_loc.convert('RGB')
    return img_loc












