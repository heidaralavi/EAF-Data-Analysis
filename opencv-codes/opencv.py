import os
import numpy as np
import pandas as pd
import cv2 
working_dir = os.getcwd()
#working_dir ='..'  # Use on Jupyter Notebook

img = cv2.imread(f"{working_dir}/opencv-images/isotermal-b3-1_5.jpg", cv2.IMREAD_GRAYSCALE)
(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


#cv2.imwrite(f"{working_dir}/opencv-images/1.png", im_bw)
#cv.imshow("Display window", img)
#k = cv.waitKey(0) # Wait for a keystroke in the window
#cv.imwrite('image1.png', img)
print(im_bw[:,0].shape)
dic = {}
for i in range(522):
    dic[i] = im_bw[:,i]
df = pd.DataFrame(dic)
df.drop(index=[0,1,2,3,4,5,6,329,330,331,332,333,334],inplace=True)
df.drop(columns=[0,1,2,3,4,5,6,7,8,9,10,11,514,515,516,517,518,519,520,521],inplace=True)
df.reset_index(drop=True,inplace=True)
df.columns = range(df.shape[1])
print(df.shape)
df.to_csv(f"{working_dir}/opencv-images/1.csv", index=True,header=True)

dicxy = {}
xlist = []
ylist = []
label = []
for i in range(502):
    for j in range(322):
        if df.iloc[j,i]== 255:
            xlist.append(i)
            ylist.append(j)
            label.append(1)
        if df.iloc[j,i]== 0:
            xlist.append(i)
            ylist.append(j)
            label.append(0)

dicxy['x'] = xlist
dicxy['y'] = ylist
dicxy['label'] = label
df = pd.DataFrame(dicxy)
df.to_csv(f"{working_dir}/opencv-images/1xy.csv", index=True,header=True)
#print(df.iloc[0,0])