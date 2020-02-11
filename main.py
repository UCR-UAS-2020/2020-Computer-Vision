import argparse as ap
import cv2
import numpy as np
import functions as fn

'''
    2020 AUVSI-SUAS UCR-UAS Computer Vision

    - Input:
                JPG image as numpy array
                Should be called using argparse as 'main.py -<filename>'
    - Result:
                Stores a duplicate copy of the input image in the '.\db\out' local directory
                Creates a file '*.jpg' with the same filename in the '.\db' local directory
                Creates a file '*.txt' with the same filename with the required classification data (see template.txt)
                    in the '.\db' local directory

    Pseudo-code: ('<' is assignment, def is container initialization)
        array imRGB < imread(filename)

        imRGB < clean(imRGB)                                        # smooth and prepare image

        imHSB < RGB_to_HSV(imRGB)

        def HSV_dist_max                                            # max dist from center point in HSV space for bin

        array HSV_bins, array frequencies < histogram(im)           # Array of HSV colors that are closest to rectangular blob centers (n-modal partitioning)
            # TODO: See if the signal to noise ratio is too low for good histogram

        def Thresh_max_freq                                         # If a color is too common don't use it

        for i = 1:len(frequencies)
            if freq[i] > Thresh_max_freq)
                break

            binMask < createMask (imHSV, HSV_bins[i], HSV_dist_max)
            centroids < findCentroids(binMask)
            boxBounds < createBoundingbox(binMask, center)

        array posCenters < findCenters(imHSV, colorChannel, HSV_dist_max)
        targets_cropped, ref_colors < RGB   # Regions of pixels that fall within the bounding box
'''

fp = r'./Standard Target Images 1/imrs.jpg'
im = cv2.imread(fp)  # TODO: Get argparse working
colorTol = (50, 50, 50)
tColors = fn.hsvHist(im)

# TODO: imshow debug images
for color in tColors:
    mask = fn.createColorMask(im, color, colorTol)
    centroids = fn.findCentroids(im, mask, color, colorTol)
    for centroid in centroids:
        bBox = fn.findBBox(im, mask, centroid, color, colorTol)
        crop = fn.cropRegion(im, bBox)
        # classify with mask
