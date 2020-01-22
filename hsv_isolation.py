import cv2
import numpy as np
# import matplotlib.pyplot as plt

# img = cv2.imread('group_photo.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# plt.imshow(img)


# hue, sat, val = img[:,:,0], img[:,:,1], img[:,:,2]
#
# plt.figure(figsize=(10,8))
# plt.subplot(311)                             #plot in the first cell
# plt.subplots_adjust(hspace=.5)
# plt.title("Hue")
# plt.hist(np.ndarray.flatten(hue), bins=180)
# plt.subplot(312)                             #plot in the second cell
# plt.title("Saturation")
# plt.hist(np.ndarray.flatten(sat), bins=128)
# plt.subplot(313)                             #plot in the third cell
# plt.title("Value")
# plt.hist(np.ndarray.flatten(val), bins=128)
# plt.show()

while(1):

    # Take each frame
    frame = cv2.imread("group_photo.jpg")

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)

    # define range of different colors in HSV
    # Green
    # lower_green = np.array([40, 100, 100])
    # upper_green = np.array([80, 255, 255])
    #
    # # Blue
    # lower_blue = np.array([108, 100, 100])
    # upper_blue = np.array([148, 255, 255])

    # Red (Pinkish-Red)
    # lower_red = np.array([235, 100, 100])
    # upper_red = np.array([255, 255, 255])

    #Yellow
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Threshold the HSV image to get only colors
    # Green mask
    # mask = cv2.inRange(hsv, lower_green, upper_green)
    #
    # # Blue mask
    # mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    # Red mask
    # mask3 = cv2.inRange(hsv, lower_red, upper_red)

    # Yellow mask
    mask4 = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise-AND mask and original image
    # Green result
    # greenRes = cv2.bitwise_and(frame, frame, mask = mask)
    #
    # # Blue result
    # blueRes = cv2.bitwise_and(frame, frame, mask = mask2)

    # Red result
    # redRes = cv2.bitwise_and(frame, frame, mask = mask3)

    # Yellow result
    yellowRes = cv2.bitwise_and(frame, frame, mask = mask4)

    # Bitwise-OR the results to get the final product
    # greenBlueRes = cv2.bitwise_or(greenRes, blueRes)
    # res = cv2.bitwise_or(greenBlueRes, redRes)

    # Display result
    cv2.imshow('frame',frame)
    cv2.imshow('res', yellowRes)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()