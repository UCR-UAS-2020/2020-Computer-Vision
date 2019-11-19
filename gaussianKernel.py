import numpy as np
import math

def gaussKern(size, sigma):
    output = []         # The 2D array that represents the Guassian Kernel.

    # Dynamically populates the 2D array with the result of applying the 2D Gaussian function to each index in the
    # kernel.
    for r in range(1, size + 1):
        row = []    # Holds the values for each element in a row of size 'size'.

        # Populates the row array with the 2D Gaussian function results.
        for c in range(1, size + 1):
            arrElem = (1/((2 * math.pi)*(sigma**2))) * math.exp(-((r**2+c**2)/(2*(sigma**2))))
            row.insert(c, arrElem)

        # Insert the row array into the output array.
        output.insert(r, row)

    return output

# Test case.
gKern = gaussKern(5, 2.5)

# Print the kernel just to check the values.
for r in gKern:
    for c in r:
        print(c)
    print()
