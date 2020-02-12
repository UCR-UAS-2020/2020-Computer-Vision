import sys
import cv2
from scipy.cluster.vq import vq, kmeans
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time as t

np.set_printoptions(threshold = sys.maxsize)

pil_image = Image.open('Standard Target Images 1/IMG_0602_Screenshot.png').convert('RGB')
quantized_image = pil_image.quantize(16)
plt.imshow(quantized_image)
plt.show()

image = numpy.array(quantized_image)
image = image[:, :, ::-1].copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

hue, saturation, value = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

plt.figure(figsize = (10, 8))
plt.subplots_adjust(hspace = .5)

plt.subplot(311)
plt.title("Hue")
plt.hist(np.ndarray.flatten(hue), bins = 128)

plt.subplot(312)
plt.title("Saturation")
plt.hist(np.ndarray.flatten(saturation), bins = 128)

plt.subplot(313)
plt.title("Value")
plt.hist(np.ndarray.flatten(value), bins = 128)

file = open("output.txt", "w")
file.write(str(hue))
file.close()
