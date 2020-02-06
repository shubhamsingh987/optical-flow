import cv2
import numpy as np

capture = cv2.VideoCapture(0)

# History, Threshold, DetectShadows 
# fgbg = cv2.createBackgroundSubtractorMOG2(50, 200, True)
fgbg = cv2.createBackgroundSubtractorMOG2(300, 400, True)

# Keeps track of what frame we're on
frameCount = 0

while(1):

	ret, frame = capture.read() # capture every frame
	frameCount += 1   # count frame elasped
	resizedFrame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50) # scale down the frame to increase perfomance
	fgmask = fgbg.apply(resizedFrame) #apply 
	count = np.count_nonzero(fgmask)  # count pixel changes

	if (frameCount > 10 and count > 600): # tune the sensitivity 
		cv2.putText(resizedFrame, 'cheating', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

	cv2.imshow('Frame', resizedFrame) # actual footage
	cv2.imshow('Mask', fgmask)     # pixel changes


	k = cv2.waitKey(1) & 0xff #press esc to quit
	if k == 27:
		break

capture.release()
cv2.destroyAllWindows()
