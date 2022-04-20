# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import cv2,os,urllib.request
import numpy as np
from django.conf import settings
# face_detection_videocam = cv2.CascadeClassifier(os.path.join(
# 			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# face_detection_webcam = cv2.CascadeClassifier(os.path.join(
# 			settings.BASE_DIR,'opencv_haarcascade_data/haarcascade_frontalface_default.xml'))
# # load our serialized face detector model from disk
# prototxtPath = os.path.sep.join([settings.BASE_DIR, "face_detector/deploy.prototxt"])
# weightsPath = os.path.sep.join([settings.BASE_DIR,"face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# maskNet = load_model(os.path.join(settings.BASE_DIR,'face_detector/mask_detector.model'))
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time



# class VideoCamera(object):
# 	def __init__(self):
# 		self.video = cv2.VideoCapture(0)

# 	def __del__(self):
# 		self.video.release()

# 	def get_frame(self):
# 		success, image = self.video.read()
# 		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
# 		# so we must encode it into JPEG in order to correctly display the
# 		# video stream.

# 		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# 		for (x, y, w, h) in faces_detected:
# 			cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
# 		frame_flip = cv2.flip(image,1)
# 		ret, jpeg = cv2.imencode('.jpg', frame_flip)
# 		return jpeg.tobytes()


# class IPWebCam(object):
# 	def __init__(self):
# 		self.url = "http://192.168.0.100:8080/shot.jpg"

# 	def __del__(self):
# 		cv2.destroyAllWindows()

# 	def get_frame(self):
# 		imgResp = urllib.request.urlopen(self.url)
# 		imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
# 		img= cv2.imdecode(imgNp,-1)
# 		# We are using Motion JPEG, but OpenCV defaults to capture raw images,
# 		# so we must encode it into JPEG in order to correctly display the
# 		# video stream
# 		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 		faces_detected = face_detection_webcam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
# 		for (x, y, w, h) in faces_detected:
# 			cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
# 		resize = cv2.resize(img, (640, 480), interpolation = cv2.INTER_LINEAR) 
# 		frame_flip = cv2.flip(resize,1)
# 		ret, jpeg = cv2.imencode('.jpg', frame_flip)
# 		return jpeg.tobytes()


class MaskDetect(object):
	def __init__(self):
		self.vs = VideoStream(src=0).start()

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
		image = self.vs.read()
		image = imutils.resize(image, width=650)
		image = cv2.flip(image, 1)
		mp_drawing = mp.solutions.drawing_utils # Drawing helpers
		mp_holistic = mp.solutions.holistic # Mediapipe Solutions
		name = "TadaAsana"
		acc = []
		capture_duration = 10
		# cap = cv2.VideoCapture(0)
		# Initiate holistic model
		with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

			#start_time = time.time()
			
				
				# Recolor Feed
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				image.flags.writeable = False        
				
				# Make Detections
				results = holistic.process(image)
			
				
				# Recolor image back to BGR for rendering
				image.flags.writeable = True   
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
				

				# 4. Pose Detections
				mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
										mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
										mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
										)

			
				# Export coordinates
				try:

					
					# Extract Pose landmarks
					pose = results.pose_landmarks.landmark
					pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
					
					
					# Concate rows
					row = pose_row

					X = pd.DataFrame([row])
					body_language_class = model.predict(X)[0]
					body_language_prob = model.predict_proba(X)[0]
					#print(body_language_class, body_language_prob)


					if body_language_class == name:
						# Get status box
						cv2.rectangle(image, (0,0), (300, 100), (245, 117, 16), -1)
						
						# Display Class
						cv2.putText(image, 'CLASS'
									, (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
						cv2.putText(image, body_language_class.split(' ')[0]
									, (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
						
						'''  # Display Probability
						cv2.putText(image, 'PROB'
									, (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
						cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
									, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),  2, cv2.LINE_AA)
						
						
						#print(acc[-1])'''
						acc.append(str(round(body_language_prob[np.argmax(body_language_prob)],2)*100))


					else:

						cv2.putText(image, 'Please Perform the pose'
									, (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
						


				
				except:
					pass
				image
								
				# cv2.imshow('Raw Webcam Feed', jpeg)
				

		# 		if cv2.waitKey(10) & 0xFF == ord('q'):
		# 			break

		# cap.release()
		# cv2.destroyAllWindows()

		# print(acc[-1])

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		# (locs, preds) = self.detect_and_predict_mask(frame, faceNet, maskNet)

		# # loop over the detected face locations and their corresponding
		# # locations
		# for (box, pred) in zip(locs, preds):
		# 	# unpack the bounding box and predictions
		# 	(startX, startY, endX, endY) = box
		# 	(mask, withoutMask) = pred

		# 	# determine the class label and color we'll use to draw
		# 	# the bounding box and text
		# 	label = "Mask" if mask > withoutMask else "No Mask"
		# 	color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# 	# include the probability in the label
		# 	label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# 	# display the label and bounding box rectangle on the output
		# 	# frame
		# 	cv2.putText(frame, label, (startX, startY - 10),
		# 				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		# 	cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		ret, image = cv2.imencode('.jpg', image)
		return image.tobytes()
		
# class LiveWebCam(object):
# 	def __init__(self):
# 		self.url = cv2.VideoCapture("rtsp://admin:Mumbai@123@203.192.228.175:554/")

# 	def __del__(self):
# 		cv2.destroyAllWindows()

# 	def get_frame(self):
# 		success,imgNp = self.url.read()
# 		resize = cv2.resize(imgNp, (640, 480), interpolation = cv2.INTER_LINEAR) 
# 		ret, jpeg = cv2.imencode('.jpg', resize)
# 		return jpeg.tobytes()
