'''
    A bunch of stuff and things to build things and stuff idk
'''
import cv2
import numpy

'''
    function hsvHist(im)
    
    Search the color space for likely target candidate colors based on their relative frequency.

    Params:
        arr im: the image as openCV image
    Return:
        arr colors: an array of likely target colors
'''
def hsvHist(im):
    return


'''
    function createColorMask(im, color, colorTol)
    
    Return a binary mask of all pixels that are near the target color
    Params:
        arr im: the image as openCV image
        arr color: a 3x1 array specifying the color being searched for
        colorTol: a 3x1 array specifying the distance each color channel was allowed to differ from the main color. (use absolute difference)
    Return:
        arrMask: the required mask
'''
def createColorMask(im, color, colorTol):
    return



'''
    def findCentroids(im,mask,color,colorTol)

    Locate the centroids of small, isolated blobs of pixels.
    
    Params:
        arr im: the image as openCV image
        arr mask: a binary mask of the image under filtered using color and colorTol
        arr color: a 3x1 array specifying the color being searched for
        colorTol: a 3x1 array specifying the distance each color channel was allowed to differ from the main color. (use absolute difference)
    Return:
        arr centroids: an nx2 array of coordinates locating centroids.
'''
def findCentroids(im, mask, color, colorTol):
    return



'''
    Create a bounding box for the blob with a given center and with the specified color. The color may vary within a
    specified tolerance.
    
    Note:
    
    Params:
        arr im: the image as openCV image
        arr mask: a binary mask of the image under filtered using color and colorTol
        arr color: a 3x1 array specifying the color being searched for
        colorTol: a 3x1 array specifying the distance each color channel was allowed to differ from the main color. (use absolute difference)
    Return:
        arr bbox: a 4x1 array specifying the location of the top left pixel (x,y), height h and width w of the bounding 
        box as bbox = [x, y, w, h] 
'''
def findBBox(im, mask, centroid, color, colorTol):
    return

'''
    function cropRegion

    Splice a rectangular region from the image using bounding box coordinates.

    Params:
        arr im: the image as openCV image
        arr bBox: the bounding box as bbox = [x, y, w, h]
'''
def cropRegion(im, bBox):
    return