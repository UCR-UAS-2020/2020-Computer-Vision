import cv2
import numpy as np
import matplotlib.pyplot as plt

run = True

while (1):

    # Take each frame
    image = cv2.imread("IMG_0602.JPG")

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if run:
        frequencies = []

        for num in range(255):
            counter = 0
            lower = np.array([num, 100, 100])
            upper = np.array([num, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            for number in np.nditer(mask):
                if (number > 0):
                    counter += 1
            frequencies.append(counter)


        x = np.arange(255)

        plt.plot(x, frequencies)
        plt.show()
        print(frequencies)

        # Bitwise-AND mask and original image
        # Green result
        # greenRes = cv2.bitwise_and(frame, frame, mask = mask)
        #
        # # Blue result
        # blueRes = cv2.bitwise_and(frame, frame, mask = mask2)

        # Red result
        # redRes = cv2.bitwise_and(frame, frame, mask = mask3)

        # Yellow result
        # yellowRes = cv2.bitwise_and(image, image, mask = mask4)

        # Bitwise-OR the results to get the final product
        # greenBlueRes = cv2.bitwise_or(greenRes, blueRes)
        # res = cv2.bitwise_or(greenBlueRes, redRes)

        run = False

    # Display result
    cv2.imshow('frame', image)
    # cv2.imshow('res', yellowRes)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
