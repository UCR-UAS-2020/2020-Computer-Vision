from enum import Enum
import numpy as np


class ScaleSpace(Enum):
    o = 0
    s = 1
    x = 2
    y = 3
    theta = 4


# simulates having a 5-wide array like the one we are using to store keypoints
arr = np.arange(start=0, stop=20, dtype=int).reshape(4, 5)
print(arr)
# formatting is as follows to access enum:
# ssp_e[keypoint, ScaleSpace.index.'value']
print(arr[0, ScaleSpace.y.value])
