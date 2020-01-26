import cv2
import numpy as np
import matplotlib.pyplot as plt

run = True

while (1):

    # Take each frame
    image = cv2.imread("group_photo.jpg")

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if run:
        frequencies = []

        counter = 0
        lower = np.array([0, 100, 100])
        upper = np.array([15, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([15, 100, 100])
        upper = np.array([31, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([31, 100, 100])
        upper = np.array([47, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([47, 100, 100])
        upper = np.array([63, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([63, 100, 100])
        upper = np.array([79, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([79, 100, 100])
        upper = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([95, 100, 100])
        upper = np.array([111, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([111, 100, 100])
        upper = np.array([127, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([127, 100, 100])
        upper = np.array([143, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([143, 100, 100])
        upper = np.array([159, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([159, 100, 100])
        upper = np.array([175, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([175, 100, 100])
        upper = np.array([191, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([191, 100, 100])
        upper = np.array([207, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([207, 100, 100])
        upper = np.array([223, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([223, 100, 100])
        upper = np.array([239, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)
        counter = 0

        counter = 0
        lower = np.array([239, 100, 100])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        for num in np.nditer(mask):
            if (num > 0):
                counter += 1
        frequencies.append(counter)

        x = np.arange(16)

        plt.plot(x, frequencies)
        plt.show()

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
