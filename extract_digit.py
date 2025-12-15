import cv2

img = cv2.imread("C:/Users/Airtour/Desktop/images.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#آستانه گذاری
T, threshImg = cv2.threshold(gray, 150, 200, cv2.THRESH_BINARY_INV)

cv2.imshow("image", threshImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
