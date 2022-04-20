from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)


import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time




mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
name = "TadaAsana"
acc = []
capture_duration = 10
cap = cv2.VideoCapture(0)

print("cap type is ",type(cap))
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    #start_time = time.time()
    
    while cap.isOpened(): #and (int(time.time() - start_time) < capture_duration ):
        ret, frame = cap.read()
        print("frame type is ",type(frame))
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        jpeg = image
                        
        cv2.imshow('Raw Webcam Feed', jpeg)
        

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(acc[-1])