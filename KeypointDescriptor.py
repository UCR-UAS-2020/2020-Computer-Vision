import cv2 as cv
import numpy as np
from ExtremaFilter import findExtrema

# this is the test 'images', these will be replaced by the DoGs if this is implemented
# 'img' is the center DoG that we check if it is an extremum
zero = cv.imread(r'.\noise1.jpg', cv.IMREAD_GRAYSCALE)
img = cv.imread(r'.\noise2.jpg', cv.IMREAD_GRAYSCALE)
two = cv.imread(r'.\noise3.jpg', cv.IMREAD_GRAYSCALE)
extremas = findExtrema(zero, img, two)

img = cv.copyMakeBorder(img, top=8, bottom=8, left=8, right=8, borderType=cv.BORDER_REPLICATE)


# todo; it's at this point where it breaks: [x = 257, y = 89. extremas[9]
# extrema_set is formatted in [coord_num, column_num, row_num]
def KeypointDescriptor(extrema_set):
    extrema_set += 8
    descriptors = np.zeros([extrema_set.shape[0], 4, 4, 8])
    # todo: figure out the ValueError: could not broadcast input array from shape (5,3) into shape (5,5)
    for keypoints in range(0, extrema_set.shape[0]):
        # divides the given image into quadrants, assumes that both dimensions are even-numbered
        # creates pad on the bottom and right edges so that derivatives may be taken
        # zones - [group_row, group_column, inner_row, inner_column]
        zones = np.zeros([4, 4, 5, 5], dtype=int)
        for j in range(0, 4):
            for k in range(0, 4):
                v_lower_bound = extrema_set[keypoints, 1] + (- 8 + (4 * j))
                v_upper_bound = extrema_set[keypoints, 1] + (- 3 + (4 * j))
                h_lower_bound = extrema_set[keypoints, 0] + (- 8 + (4 * k))
                h_upper_bound = extrema_set[keypoints, 0] + (- 3 + (4 * k))
                temp = np.array(img[v_lower_bound:v_upper_bound,
                                    h_lower_bound:h_upper_bound])
                zones[j, k] = temp

        # this stores the magnitudes, in the form of [cartesian quadrant, closest 45 degree angle]
        keypoint_descriptor = np.zeros((4, 4, 8))
        # iterates through every point
        # todo: fix dimensions
        for group_row in range(0, zones.shape[0]):
            for group_column in range(0, zones.shape[1]):
                for y in range(0, zones.shape[2] - 1):
                    for x in range(0, zones.shape[3] - 1):
                        # compares to the pixel to the right and below, respectively derivative = [delta x, delta y]
                        derivative = np.array([zones[group_row, group_column, y, x + 1] - zones[group_row, group_column, y, x],
                                               zones[group_row, group_column, y + 1, x] - zones[group_row, group_column, y, x]])
                        # uses pythagorean's theorem to find the magnitude
                        vector_magnitude = np.hypot(derivative[0], derivative[1])
                        vector_angle = np.degrees(np.arctan2(derivative[1], derivative[0]))
                        # makes everything between 0 and 360
                        if vector_angle < 0:
                            vector_angle += 360
                        # rounds the angles to the closest 1/8 circle
                        vector_angle = ((vector_angle + 22.5) // 45)
                        # temp fix, makes sure that angle 0 goes in the right index
                        if vector_angle == 8:
                            vector_angle = 1
                        keypoint_descriptor[group_row, group_column, int(vector_angle)] += vector_magnitude
        descriptors[keypoints] = keypoint_descriptor

    # todo: get correct return statement
    return descriptors


print(KeypointDescriptor(extremas))
