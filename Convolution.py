'''
    This script contains functions for simple image convolution.
'''

import math
import numpy as np
import cv2

'''
    Convolve a target image (M by N) with a convolution and return the result as an (M by N)
'''


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * kernel).sum()

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k


'''
    Create a gaussian kernel with size 8 standard deviations, +- 4*sigma, based on the ipol paper.
'''


def gauss_kernel(sigma):
    # size is +-4 standard deviations, rounded up to the nearest odd number
    size = (sigma * 8) // 2 * 2 + 1
    arr = np.zeros((size, size))
    # calculate s, a constant dependent on sigma:
    s = 2*sigma**2
    # calculate o, the center of the kernel. This will never be half because we start counting at 0.
    o = size / 2
    for i in range(0, size):
        for j in range(0, size):
            dist = (i-o)**2 + (j-o)**2  # calculate distance from center of kernel squared
            # G(x,y,sigma) = e^(-(x^2 + y^2)/(2*sigma^2)) / (2*pi*sigma^2)
            arr[i, j] = math.exp(-dist/s) / (math.pi*s)

    return arr
