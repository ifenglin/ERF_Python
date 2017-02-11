from PIL import Image
import numpy

def readImage(path):
    im = Image.open(path)  # Can be many different formats.
    pix = im.load()
    print 'load image: {}'.format(path)
    print 'image size: {}'.format(im.size)  # Get the width and hight of the image for iterating over
    # print pix[0, 0]  # Get the RGBA Value of the a pixel of an image
    #  pix[x,y] = value # Set the RGBA Value of the image (tuple)
    return pix, im.size, im

def readImageRGB(path):
    _, im_size, im = readImage(path)
    r, g, b = im.split()
    return r.load(), g.load(), b.load(), im_size

def readLabels(path):
    return readImage(path)

def printLabels(labels):
    max_value = numpy.amax(labels)
    regular_coefficient = 255 / max_value
    comp_R = labels * regular_coefficient
    arr_R = (comp_R).astype('uint8')  # sample array
    im_R = Image.fromarray(arr_R)  # monochromatic image
    print im_R
    comp_G = 255 / labels
    arr_G = (comp_G).astype('uint8')  # sample array
    im_G = Image.fromarray(arr_G)  # monochromatic image
    print im_G
    comp_B = (labels % 3) * 127
    arr_B = (comp_B).astype('uint8')  # sample array
    im_B = Image.fromarray(arr_B)  # monochromatic image
    print im_B
    imrgb = Image.merge('RGB', (im_R, im_G, im_B))  # color image
    imrgb.show()
