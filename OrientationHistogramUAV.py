# converts the continuous angle of the gradient to a histogram bin:
def quantize_orientation(theta, num_bins): 
  	bin_width = 360//num_bins 
  	return int(np.floor(theta)//bin_width)

# need a Gaussian filter for creation of the Gaussian octave
def gaussian_filter(sigma): 
 	size = 2*np.ceil(3*sigma)+1 
  	x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1] 
  	g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
  	return g/g.sum()

# gets the gradient in polar coordinates at a pixel in L:
def get_grad(L, x, y): 
  	dy = L[min(L.shape[0]-1, y+1),x] — L[max(0, y-1),x] 
  	dx = L[y,min(L.shape[1]-1, x+1)] — L[y,max(0, x-1)] 
  	return cart_to_polar_grad(dx, dy)

# fit a parabola to the three histogram values closest to the maximum
# We simply get the least squares solution where the center of the max
# histogram bin as well as its two adjacent bins are taken as the independent
# variable and the value at that histogram the dependent variable. Once the 
# coefficients for the parabola have been found, just use -b/2a to get the 
# refined orientation.
def fit_parabola(hist, binno, bin_width): 
  	centerval = binno*bin_width + bin_width/2.
 
  	if binno == len(hist)-1: 
		rightval = 360 + bin_width/2. 
  	else: 
		rightval = (binno+1)*bin_width + bin_width/2.
 
  	if binno == 0: 
		leftval = -bin_width/2. 
  	else: 
		leftval = (binno-1)*bin_width + bin_width/2.
 
  	A = np.array([[centerval**2, centerval, 1], [rightval**2, rightval, 1], [leftval**2, leftval, 1]]) 
  	b = np.array([hist[binno], hist[(binno+1)%len(hist)], hist[(binno-1)%len(hist)]]) 
  	x = LA.lstsq(A, b, rcond=None)[0] 

  	if x[0] == 0: 
		x[0] = 1e-6 
  	return -x[1]/(2*x[0])

def assign_orientation(kps, octave, num_bins=36):
	# array of keypoints 
	new_kps = [] 
	# histogram created on angle with 36 bins each with width of 10 degress (360 degrees)
  	bin_width = 360
	# iterate through each kepoint in the keypoint array 
  	for kp in kps: 
		# convert first three points in array to integers
    		cx = int(kp[0]) 
		cy = int(kp[1])
		s = int(kp[2]) 
		# given interval, values outside the interval are clipped to the interval edges
		# meaning values samller than the min are set to the min and the values greater 
		# than max are set to max
    		s = np.clip(s, 0, octave.shape[2]-1) 
		# weight eahc sample added by its gradient magnitude and by a Gaussian-weighted 
		# circular window with a σ that is 1.5 times that of the scale of the keypoint.
    		sigma = kp[2]*1.5 
		# ceiling of the input, element-wise 
		# the ceil of the scalar x is the smallest integer i, such that i >= x 
    		w = int(2*np.ceil(sigma)+1) 
		# guassian filter given standard deviation for guassian kernel
    		kernel = gaussian_filter(sigma) 
		# an octave is actually a set of images were the blur of the last image is double
		# the blur of the first image. S is the number of images we want in each octave
    		L = octave[…,s]
		# create histogram with an array of given shape and type, filled with zeros
    		hist = np.zeros(num_bins, dtype=np.float32) 
	# the histogram is created on angle (the gradient is specified in polar coordinates)
	# and has 36 bins (each bin has a width of 10 degrees). When the magnitude and angle
	# of the gradient at a pixel is calculated, the corresponding bin in our histogram 
	# grows by the gradient magnitude weighted by the Gaussian window.
	for oy in range(-w, w+1): 
		for ox in range(-w, w+1): 
      	  		x = cx+ox 
			y = cy+oy 
        		if x < 0 or x > octave.shape[1]-1: 
				continue 
        		elif y < 0 or y > octave.shape[0]-1: 
				continue 

        		m = get_grad(L, x, y)
			theta = get_grad(L, x, y) 
        		weight = kernel[oy+w, ox+w] * m 
        		bin = quantize_orientation(theta, num_bins) 
        		hist[bin] += weight 
	# we assign that keypoint the orientation of the maximal histogram bin.
    	max_bin = np.argmax(hist) 
    	new_kps.append([kp[0], kp[1], kp[2], fit_parabola(hist, max_bin, bin_width)]) 
    	max_val = np.max(hist) 
    	for binno, val in enumerate(hist): 
      		if binno == max_bin: continue 
      			if .8 * max_val <= val: 
        			new_kps.append([kp[0], kp[1], kp[2], fit_parabola(hist, binno, bin_width)])

	return np.array(new_kps)