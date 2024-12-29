import cv2 as cv
img = cv.imread("/app/opencv-images/photo15978847178.jpg")

#cv.imshow("Display window", img)
#k = cv.waitKey(0) # Wait for a keystroke in the window
#cv.imwrite('image1.png', img)
print(img.size)