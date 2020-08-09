import cv2
face_cascade = cv2.CascadeClassifier('/home/disha/opencv/data/haarcascades/haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('/home/disha/opencv/data/haarcascades/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('/home/disha/node-opencv/data/haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('/home/disha/node-opencv/data/haarcascade_mcs_mouth.xml')

cap = cv2.VideoCapture(0)
while 1:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
  
    for (x,y,w,h) in faces: 
        # To draw a rectangle in a face  
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w]
        
        # Detects eyes,nose,mouth of different sizes in the input image 
        eyes = eye_cascade.detectMultiScale(roi_gray,1.2,3)
        nose = nose_cascade.detectMultiScale(roi_gray,1.1,1)
        mouth = mouth_cascade.detectMultiScale(roi_gray,1.2,3)
  
        #To draw a rectangle in eyes,nose,mouth
        for (ex,ey,ew,eh) in eyes: 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        for (nx,ny,nw,nh) in nose:
            cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
        for (mx,my,mw,mh) in mouth:
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(255,0,255),2)
  
    # Display an image in a window 
    cv2.imshow('Face detection',img) 
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break
  
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  
