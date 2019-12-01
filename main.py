# loaded modules:
from skimage.exposure import rescale_intensity
import math
import numpy as np
# import argparse
import cv2

# project files:
from Convolution import convolve
from Convolution import gauss_kernel
from ExtremaFilter import findExtrema
from KeypointLocalization import get_keypoints

def scale_image(im, scale):
    dim = (int(im.shape[1]*scale) , int(im.shape[0]*scale))
    return cv2.resize(test_im, dim, interpolation=cv2.INTER_AREA)

# parameters:
sigma_min = 0.8                 #
sigma_in = 0.5                  #
n_spo = 5                       # Number of scales in an octave
n_o = 3                         # Number of octaves
delta_min = 0.5

C_contrast = 0.03
C_edge = 0.1

len_in = 600                    # rescale length for large images. Height is scaled to preserve aspect ratio.

# TODO: scale to width 600 regardless of input size in case we mess up camera settings on accident

# BEGIN:

# input test image as test_im
test_im = cv2.imread(
    r'C:\Users\chris\Documents\GitHub\2020-Computer-Vision\Standard Target Images 1\Test Images\DSC00099.jpg',
    cv2.IMREAD_GRAYSCALE)
#cv2.imshow('Input Image', test_im)

# generate parameters for first octave

# subsample original image (high resolution)
# TODO: Update subsampling resolution for new input image format

test = test_im.shape

im1 = scale_image(test_im, .10)
# scale by assumed inter pixel width of input image
im1 = scale_image(im1, 1/delta_min)
# take height and width for later
(iH, iW) = im1.shape[:2]

# create scale space array
# ssp: a scale space array of the original test image. Indices are:
# the number of the octave,
# the number of the scale in the octave,
# the column number,
# the row number
# there are 2 extra scales for later on in the algorithm
ssp = np.zeros((n_o, n_spo+3, iH, iW))
# set first sigma as sigma min
sigma = sigma_min

# compute the first gaussian
new_gauss = convolve(im1, gauss_kernel(sigma))
old_gauss = np.zeros(new_gauss.shape)
# compute the first octave
for s in range(0, n_spo+2):
    sigma = sigma*2**(1/n_spo)
    old_gauss = new_gauss
    new_gauss = convolve(im1, gauss_kernel(sigma))
    ssp[0, s] = np.subtract(new_gauss, old_gauss)
    print('Completed: o=' + str(0) + '\ts=' + str(s))
# compute the rest of the octaves
for o in range(1, n_o):
    # compute the first gaussian as the half scale of the end of the last octave
    for i in range(0, iH):
        for j in range(0, iW):
            new_gauss[i, j] = old_gauss[(i//2)*2, (j//2)*2]
    # compute the other DoGs in the octave
    for s in range(1, n_spo+2):
        sigma = sigma*2**(1/n_spo)
        old_gauss = new_gauss
        new_gauss = convolve(im1, gauss_kernel(sigma))
        ssp[o, s] = np.subtract(new_gauss, old_gauss)
        print('Completed: o=' + str(o) + '\ts=' + str(s))

# Filter Extrema

# create an array to track keypoint candidates with indices (o, s, y, x) where
# TODO explain indices

ssp_e = np.empty((0, 4))
for o in range(0, n_o):
    for s in range(1, n_spo + 1):
        extrema = findExtrema(ssp[o, s], ssp[o, s+1], ssp[o, s+2])
        # create an array indicating octave and scale in the same shape as extrema
        # TODO: Actually make this work. Need to check on indices
        os = [o, s]
        os = np.repeat(os, extrema.shape[1], axis=1)
        extrema = np.hstack(os, extrema)
        np.append(extrema, ssp_e)

temp = np.empty((0, 4))
# Low Contrast Filter
for w in ssp_e:
    if ssp[w] >= C_contrast * .8:
        np.append(temp, w)


get_keypoints(ssp_e, 16)


# Create gaussian blur convolutions for scale space

# im_size
#   a 1x2 array of the width and height as [w,h] of the input image

# Scale_Initial
# Scale_Final
# Scale_number

# Octave_Initial
# Octave_Final
# Octave_Number


# apply DOG iteratively, and all of the results in an array

#   The array will be 4D and have the indices: Octave Scale X Y
# Scan for local extrema using nearest neighbors
#   im_array_temp
#       A temporary array for the nearest neighbor search
# Localization using taylor polynomial
#   Cull some maxima
# Hough transform for edge responses
# Assign orientation
# Match to database

