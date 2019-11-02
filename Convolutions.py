# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def convolve(image, kernel):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
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
            k = (roi * kernel).sum() / 159

            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k

    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")

    # return the output image
    return output

kernel = np.array((
    [0, 1, 0],
    [1, 4, 1],
    [0, 1, 0]))

kernel1 = np.array((
    [2, 4,  5,   4, 2],
    [4, 9,  12,  9, 4],
    [5, 12, 15, 12, 5],
    [4,  9, 12,  9, 4],
    [2,  4,  5,  4, 2]
))
img1 = cv2.imread(r'C:\Users\chris\Desktop\Labrador-Retriever-MP.jpg',0)

img2 = convolve(img1, kernel1)

img12 = np.hstack((img1,img2))

cv2.imshow('lol', img12)
cv2.waitKey()