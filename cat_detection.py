#detecting cat 
# with help of (Py)imageSearch
# import the necessary packages

import argparse
import cv2


##construct the argument prase
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, 
	help = "path to the input image")
ap.add_argument("-c", "--cascade", 
	default = "haarcascade_frontalcatface_extended.xml",
	help = "path to cat detector haar cascade")
args = vars(ap.parse_args())

##load the input image
image = cv2.imread(args["image"])
##Converting image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

##load the cat detector haar cascade, then detectc cat faces
detector = cv2.CascadeClassifier(args["cascade"])
##Change scaleFactor as such it is > 1 and the more the scale factor the lower the results.
##the lesser the scaleFactor the longer it will take to process.
rects = detector.detectMultiScale(gray,scaleFactor = 1.2,
	minNeighbors = 10, minSize = (50,50))
	

##loop over the cat faces and draw a rectangle surrounding each
for (i,(x,y,w,h)) in enumerate(rects):
	cv2.rectangle(image,(x,y), (x+w,y+h),(0,0,255),2)
	cv2.putText(image,"Cat #{}".format(i+1),(x,y-10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55,(0,0,255),2)

## displat of the final image
cv2.imshow("cat faces", image)
## the image will be shown for 7000 miliseconds or 7 seconds.
## if you want to see the image indefinitly than put 0 instead of 7000
cv2.waitKey(7000)
