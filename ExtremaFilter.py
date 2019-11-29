import numpy as np
import cv2 as cv

# this is the test 'images', these will be replaced by the DoGs if this is implemented
# 'img' is the center DoG that we check if it is an extremum
zero = cv.imread(r'.\noise1.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread(r'.\noise2.jpg', cv.IMREAD_GRAYSCALE)
two = cv.imread(r'.\noise3.jpg', cv.IMREAD_GRAYSCALE)


# parameters: the 3 different DoGs
def findExtrema(top, mid, bot):
    # add borders around each image
    top = cv.copyMakeBorder(top, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    mid = cv.copyMakeBorder(mid, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
    bot = cv.copyMakeBorder(bot, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)

    # creates array for extrema to be stored
    coords = np.empty((0, 2))
    coords_height = 0
    # puts the 3 DoG's from the octave into a 3d matrix
    stack = np.stack((top, mid, bot))
    # stores the dimensions of the DoG
    image_height = mid.shape[0]
    image_width = mid.shape[1]
    filtered_image = np.zeros((image_height, image_width))

    # goes through every pixel
    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):

            # creates the 3x3x3 array in which the program checks for extrema
            temp_array = np.array(stack[0:3, y - 1:y + 2, x - 1:x + 2])
            unique_values, indices_list = np.unique(temp_array, return_counts=True)
            # looks if the max in the 3x3x3 area is the middle DoG and the same x & y coordinate
            if np.max(temp_array) == stack[1, y, x] and indices_list[-1] == 1:
                # outputs the coordinates
                filtered_image[y, x] = 255
                coords_height += 1
                # corrects for the pad - since pad moves all the numbers down and right by one,
                # we move it back up and left by one to account for it
                coords = np.append(coords, [int(x) - 1, int(y) - 1]).reshape(coords_height, 2)
                coords = coords.astype(dtype=int)
                x += 1
    return coords


# returns [x,y] coords of each extrema
'''extrema_set = findExtrema(zero, img, two)
print(extrema_set)

cv.imshow('hello', img)
cv.waitKey(0)
cv.destroyAllWindows()'''
