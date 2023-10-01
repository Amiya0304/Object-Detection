import cv2
import numpy as np
from gtts import gTTS
import os

cap=cv2.VideoCapture(0)
ret,frame1=cap.read()
ret,frame2=cap.read()
while cap.isOpened():
    fh = open("coco.names.txt", "r")
    #mytext = fh.read().replace("\n", " ")
    language= 'en'
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 225, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 700:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 225,0), 2)
        cv2.putText(frame1, "object detected".format('coco.names.txt'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 3)
        #cv2.drawContours(frame1, contours, -1, (0, 225, 0), 2)
        output=gTTS(text='object detected', lang=language, slow=False)
        output.save("output.mp3")
       
    
    cv2.imshow("feed",frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40)==27:
        fh.close()
        os.system("start output.mp3")
        break
cv2.destroyAllWindows()
cap.release()
