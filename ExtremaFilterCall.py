from ExtremaFilter import findExtrema
import cv2 as cv
import numpy as np

zero = cv.imread(r'.\noise1.jpg', cv.IMREAD_GRAYSCALE)
one = cv.imread(r'.\noise2.jpg', cv.IMREAD_GRAYSCALE)
two = cv.imread(r'.\noise3.jpg', cv.IMREAD_GRAYSCALE)
three = cv.imread(r'.\noise4.jpg', cv.IMREAD_GRAYSCALE)
four = cv.imread(r'.\noise5.jpg', cv.IMREAD_GRAYSCALE)
iH = one.shape[0]
iW = one.shape[1]
ssp = np.zeros((1, 5, iH, iW))  # o, s, y, x
ssp[0, 0] = zero
ssp[0, 1] = one
ssp[0, 2] = two
ssp[0, 3] = three
ssp[0, 4] = four

n_o = 1
n_spo = 5
ssp_e = np.empty((0, 4))
for o in range(0, n_o):
    # the upper bound for range didn't seem to work unless it was n_spo - 2, but it makes sense that it's that.
    # might need to change the main file to this later on
    for s in range(0, n_spo - 2):
        extrema = findExtrema(ssp[o, s], ssp[o, s+1], ssp[o, s+2])
        # TODO: check if there needs to be a fifth index yet (theta), but it should be easy to implement
        # create an array indicating octave and scale in the same shape as extrema
        temp_os = [o, s]
        os = np.tile(temp_os, (extrema.shape[0], 1))
        # used np.tile in attempts to create the o and s columns efficiently
        # not sure if this is faster than np.repeat or np.vstack-ing a bunch of times
        extrema = np.hstack((os, extrema))
        # TODO: np.vstack and np.hstack seem to require 2 sets of parentheses, but current main.py has only 1 set per
        #   this might be something we have to look at
        ssp_e = np.vstack((ssp_e, extrema))
        # attaches the set of extrema from this scale & octave to the main array


# for testing purposes - delete when implemented
print(ssp_e)
