import numpy as np
import numpy.linalg as linalg

def localize_keypoint(D, x, y, sigma):
    # First derivatives
    # NOTE: might need some work
    dx = (D[y, x + 1, sigma] - D[y, x - 1, sigma]) / 2
    dy = (D[y + 1, x, sigma] - D[y - 1, x, sigma]) / 2
    ds = (D[y, x, sigma + 1] - D[y, x, sigma - 1]) / 2

    # Second derivatives
    # NOTE: might need some work
    dxx = D[y, x + 1, sigma] - 2 * D[y, x, sigma] + D[y, x - 1, sigma]
    dxy = ((D[y + 1, x + 1, sigma] - D[y + 1, x - 1, sigma]) - (D[y - 1, x + 1, sigma] - D[y - 1, x - 1, sigma])) / 4
    dxs = ((D[y, x + 1, sigma + 1] - D[y, x - 1, sigma + 1]) - (D[y, x + 1, sigma - 1] - D[y, x - 1, sigma - 1])) / 4
    dyy = D[y + 1, x, sigma] - 2 * D[y, x, sigma] + D[y - 1, x, sigma]
    dys = ((D[y + 1, x, sigma + 1] - D[y - 1, x, sigma + 1]) - (D[y + 1, x, sigma - 1] - D[y - 1, x, sigma - 1])) / 4
    dss = D[y, x, sigma + 1] - 2 * D[y, x, sigma] + D[y, x, sigma - 1]

    # Jacobian matrix = matrix of first derivatives
    jacobian = np.array([dx, dy, ds])

    # Hessian matrix = matrix of second derivatives
    hessian = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])

    # Offset = -jacobian / hessian
    offset = -linalg.inv(hessian).dot(jacobian)

    # Return values
    return offset, jacobian, hessian[:2, :2], x, y, sigma

def get_candidate_keypoints(D, w=16):
    # Array of candidates
    candidates = []

    # i = x, j = y, k = z
    # Scan three levels into the difference of gaussian pyramid and add to the candidate array if the maximum/minimum of the window is 13
    for i in range(w // 2 + 1, D.shape[0] - w // 2 - 1): # D.shape[0] = rows of D
        for j in range(w // 2 + 1, D.shape[1] - w // 2 - 1): # D.shape[1] = columns of D
            for k in range(1, D.shape[2] - 1): # D.shape[2] = height
                # Get the general window of analysis
                # Look at the values around x, y, and z in the set of three scales
                window = D[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]

                # argmax gets the maximum, argmin gets the minimum, threshold value is 13
                if np.argmax(window) == 13 or np.argmin(window) == 13:
                    candidates.append([i, j, k])

    return candidates

def find_keypoints_for_octave(D, extrema_threshold, eigenvalue_ratio, width):
    # Get the possible keypoints from D, with a width
    candidates = get_candidate_keypoints(D, width)

    # Array of keypoints
    keypoints = []

    # loop through array of candidates
    for i, candidate in enumerate(candidates):
        y = candidate[0]
        x = candidate[1]
        sigma = candidate[2]

        # Get the subpixel offset, jacobian matrix, hessian matrix, and candidates from the localized keypoints
        offset, jacobian, hessian, x, y, sigma = localize_keypoint(D, x, y, sigma)

        # Contrast = D + 0.5 * (transpose of the jacobian)
        contrast = D[y, x, sigma] + 0.5 * jacobian.dot(offset)

        # if the absolute value of the contrast is greater than some experimentally determined eigenvalue ratio then continue with the loop
        if abs(contrast) > extrema_threshold: continue

        # w = array of the eigenvalues, v = normalized eigenvectors (not needed, only implemented to call function)
        w, v = linalg.eig(hessian)

        # get ratio of principal curvatures
        r = w[1] / w[0]

        # calculate ratio
        R = (r + 1) ** 2 / r

        # if ratio is less than some eigenvalue ratio, then continue
        if R < eigenvalue_ratio: continue

        # Once all checks have passed, create a keypoint value by making an array of x, y, sigma and adding the offset
        kp = np.array([x, y, sigma]) + offset

        # append keypoint
        keypoints.append(kp)

    # return array
    return np.array(keypoints)

'''
    function get_keypoints
    Inputs:
        Dog_pyr:    a scale space pyramid as a 4D array
        width:      width of a keypoint
'''


def get_keypoints(DoG_pyr, width):
    # Array of keypoints
    keypoints = []

    # Constant values for extrema threshold and eigenvalue_ration threshold
    extrema_threshold = 0.03
    eigenvalue_ratio = ((10 + 1) ^ 2) / 10

    # For every matrix of D in the Difference of Gaussian pyramid, add a new keypoint
    for D in DoG_pyr:
        keypoints.append(find_keypoints_for_octave(D, extrema_threshold, eigenvalue_ratio, width))

    # Return the array of keypoints
    return keypoints
