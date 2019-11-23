import cv2 as cv
import numpy as np

# these are purely for testing purposes: img will be the input

image = np.random.randint(low=0, high=255, size=(1, 8, 8))  # size=(height, rows, columns)


def KeypointDescriptor(img):
    # divides the given image into quadrants, assumes that both dimensions are even-numbered
    # creates pad on the bottom and right edges so that derivatives may be taken
    # i have no idea how to make this not super ugly
    first_quadrant = np.array(img[0, :img.shape[1] // 2, img.shape[2] // 2:])
    first_quadrant = cv.copyMakeBorder(src=first_quadrant, top=0, left=0, bottom=1, right=1,
                                       borderType=cv.BORDER_REPLICATE, value=0)
    second_quadrant = np.array(img[0, :img.shape[1] // 2, :img.shape[2] // 2])
    second_quadrant = cv.copyMakeBorder(src=second_quadrant, top=0, left=0, bottom=1, right=1,
                                        borderType=cv.BORDER_REPLICATE, value=0)
    third_quadrant = np.array(img[0, img.shape[1] // 2:, :img.shape[2] // 2])
    third_quadrant = cv.copyMakeBorder(src=third_quadrant, top=0, left=0, bottom=1, right=1,
                                       borderType=cv.BORDER_REPLICATE, value=0)
    fourth_quadrant = np.array(img[0, img.shape[1] // 2:, img.shape[2] // 2:])
    fourth_quadrant = cv.copyMakeBorder(src=fourth_quadrant, top=0, left=0, bottom=1, right=1,
                                        borderType=cv.BORDER_REPLICATE, value=0)

    # puts all the quadrants together to make iteration easier
    quadrants = np.array([first_quadrant,
                          second_quadrant,
                          third_quadrant,
                          fourth_quadrant])
    # this stores the magnitudes, in the form of [cartesian quadrant, closest 45 degree angle
    keypoint_descriptor = np.zeros((4, 8))
    # iterates through every point
    for z in range(0, quadrants.shape[0]):
        for y in range(0, quadrants.shape[1] - 1):
            for x in range(0, quadrants.shape[2] - 1):
                # compares to the pixel to the right and below, respectively
                derivative = [quadrants[z, y, x + 1] - quadrants[z, y, x], quadrants[z, y + 1, x] - quadrants[z, y, x]]
                # uses pythagorean's theorem to find the magnitude
                vector_magnitude = np.hypot(derivative[0], derivative[1])
                # checks if pointing straight up or down, sets angle accordingly
                if derivative[0] != 0:
                    vector_angle = np.degrees(-1 * np.arctan(derivative[1] / derivative[0]))
                else:
                    if derivative[0] > 0:
                        vector_angle = 90
                    elif derivative[0] < 0:
                        vector_angle = 270
                    else:
                        vector_angle = 0
                # points the arrow the right direction, since atan always makes it between 0 and 180
                if derivative[0] < 0:
                    vector_angle += 180
                # makes everything between 0 and 360
                if vector_angle < 0:
                    vector_angle += 360

                # rounds the angles to the closest 1/8 circle
                vector_angle = ((vector_angle + 22.5) // 45)
                # temp fix, makes sure that angle 0 goes in the right index
                if vector_angle == 8:
                    vector_angle = 1
                keypoint_descriptor[z, int(vector_angle)] += vector_magnitude
    return keypoint_descriptor


print(KeypointDescriptor(image)[0, 0])
