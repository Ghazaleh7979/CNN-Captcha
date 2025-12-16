import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
from keras.models import load_model
import numpy as np

net = load_model("digit_classifier.h5")
img = cv2.imread("C:/Users/Airtour/Desktop/images.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#آستانه گذاری
T, threshImg = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)


cnts , _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[2:]
for i in range(len(cnts)):
    x,y,w,h = cv2.boundingRect(cnts[i])
    
    roi = img[y-10:y+h+5, x-10:x+w+5]
    roi = cv2.resize(roi,(32, 32))
    roi = roi/255.0
    roi = np.array([roi])
    
    output = net.predict(roi)[0]
    max_index = np.argmax(output) + 1
    print(max_index)
    
    cv2.rectangle(img, (x-10,y-10), (x+w+5, y+h+5), (100,0,100), 2)

cv2.imshow("image", img)
cv2.waitKey(0)      
cv2.destroyAllWindows()

