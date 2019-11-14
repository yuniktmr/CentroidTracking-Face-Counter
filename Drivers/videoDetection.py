from Algorithms.centroidTracker import CentroidTracker
import cv2
import argparse
from PIL import Image
import imutils
from imutils.video import VideoStream
import numpy as np
import time 

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

ct = CentroidTracker()
(H , W) = (None, None)
print("[INFO] LOADING MODEL...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")

print("[INFO] starting video stream...")


vs = VideoStream(src = 0).start()
time.sleep(2.0)

while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width = 400)
	
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104.0, 177.0, 123.0))
	
	net.setInput(blob)
	detections = net.forward()
	
	rects = []
	
	for i in range(0, detections.shape[2]):
		
		if detections[0, 0 , i ,2] > args["confidence"]:
			box = detections[0,0,i,3:7] * np.array([W,H,W,H])
			rects.append(box.astype("int"))
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
			
	objects = ct.update(rects)
			
	for (objectID, centroid) in objects.items():
				
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]),4,(0,255,0), -1)
				
	cv2.imshow("Frame", frame)
	cv2.resizeWindow('image', 600,600)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
vs.stop()
			
		