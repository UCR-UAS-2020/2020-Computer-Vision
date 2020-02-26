import cv2


img1 = cv2.imread(r'C://Users/chris/Desktop/IMG_0602_2.JPG')
img1n = cv2.bitwise_not(img1)
img2 = cv2.imread(r'C://Users/chris/Desktop/n4.jpg')
img2n = cv2.bitwise_not(img2)
cv2.imshow('asdff',img2)
cv2.waitKey(0)
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img = cv2.drawKeypoints(gray,kp,img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('asdf',img)
cv2.waitKey(0)


# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2n,kp2,matches[:20],img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img3 = cv2.drawMatches(img1,kp1,img2n,kp2,matches[:20],img2)

cv2.imshow('asdf',img3)
cv2.waitKey(0)

