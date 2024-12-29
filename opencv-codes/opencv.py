import os
import cv2 as cv
working_dir = os.getcwd()
#working_dir ='..'  # Use on Jupyter Notebook

img = cv.imread(f"{working_dir}/opencv-images/isotermal-b3-1_5.jpg")

#cv.imshow("Display window", img)
#k = cv.waitKey(0) # Wait for a keystroke in the window
#cv.imwrite('image1.png', img)
print(img.size)