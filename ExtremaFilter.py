import numpy as np
import cv2 as cv

# this is the test 'images', these will be replaced by the DoGs if this is implemented
# 'one' is the center DoG that we check if it is an extremum
zero = cv.imread(r'.\noise1.jpg', cv.IMREAD_GRAYSCALE)
one = cv.imread(r'.\noise2.jpg', cv.IMREAD_GRAYSCALE)
two = cv.imread(r'.\noise3.jpg', cv.IMREAD_GRAYSCALE)


# parameters: the 3 different DoGs
def findExtrema(top, mid, bot):
    #add borders around each image
    top = cv.copyMakeBorder(top, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    mid = cv.copyMakeBorder(mid, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    bot = cv.copyMakeBorder(bot, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

    # puts the 3 DoG's from the octave into a 3d matrix
    stack = np.stack((top, mid, bot))

    # stores the dimensions of the DoG
    image_height = mid.shape[0]
    image_width = mid.shape[1]
    filtered_image = np.zeros((image_height, image_width))

    # goes through every pixel
    for x in range(1, image_height - 1):
        for y in range(1, image_width - 1):

            # creates the 3x3x3 array in which the program checks for extrema
            temp_array = np.array(stack[0:3, x - 1:x + 2, y - 1:y + 2])
            unique_values, indices_list = np.unique(temp_array, return_counts=True)
            # looks if the max in the 3x3x3 area is the middle DoG and the same x & y coordinate
            if np.max(temp_array) == stack[1, x, y] and indices_list[-1] == 1:
                # outputs the coordinates
                filtered_image[x, y] = 255
    return filtered_image


# these are for visualizing what the findExtrema function does - not needed when implemented into the larger project
cv.imshow('hello', findExtrema(zero, one, two))
cv.waitKey(0)
cv.destroyAllWindows()
