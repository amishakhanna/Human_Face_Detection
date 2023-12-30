# Import libraries
import cv2
import numpy as np

n=int(input("Select Option : \n 1.Face Detection From Live Video\n 2.Face Detection From Images\n"))

if(n==1) :
   # Video capture using WebCam
   cap = cv2.VideoCapture(0)
         
   # print a feedback
   print('Camera On')

   # Load face detection classifier
   # Load face detection classifier ~ Path to face & eye cascade
   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # Pre train model

   while True:
   # Original frame ~ Video frame from camera3
      ret, frame = cap.read()  # Return value (true or false) if the capture work, video frame
      
      # Convert original frame to gray
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      
      # Get location of the faces in term of position
      # Return a rectangle (x_pos, y_pos, width, height)
      faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE) 
      
      # Detect faces
      for (x, y, w, h) in faces:
         # Draw rectangle in the face
         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 53, 18), 2)  # Rect for the face
         
      # Load video frame
      cv2.imshow('Video Frame', frame)
      
      # Wait 1 millisecond second until q key is press
      # Get a frame every 1 millisecond
      if cv2.waitKey(1) == ord('q'):
         # Print feedback
         print('Camera Off')
         break
      
   # Close windows
   cap.release()  # Realise the webcam
   cv2.destroyAllWindows()  # Destroy all the windows

else:
   pixels = cv2.imread('1.jpg')
   # load the pre-trained model
   classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
   # perform face detection
   bboxes = classifier.detectMultiScale(pixels)
   # print bounding box for each detected face
   for box in bboxes:
      # extract
      x, y, width, height = box
      x2, y2 = x + width, y + height
      # draw a rectangle over the pixels
      cv2.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
   # show the image
   cv2.imshow('face detection', pixels)
   # keep the window open until we press a key
   cv2.waitKey(0)
   # close the window
   cv2.destroyAllWindows()