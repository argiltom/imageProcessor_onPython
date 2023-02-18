import sys
#print(sys.argv[0])#どうやら第零引数にはソースファイルの名前が格納されるみたい
import numpy as np
import cv2
inputFileName=sys.argv[1]
outputFileName=sys.argv[2]
img=np.float64(cv2.imread(inputFileName,-1))
#retImg=np.zeros_like(img)

intervalCount=10 #10pixelごとにα値をゼロにする．
count=0
isTransparence=False
#for y in range(img.shape[0]):
for x in range(img.shape[1]):
    if count == 10:
        count=0
        if isTransparence:
            isTransparence=False
        else:
            isTransparence= True
    if isTransparence:
        img[0:img.shape[0],x,3]=0
        print("x="+str(x))
    count+=1
cv2.imshow("ImgWindow",img)
cv2.waitKey(0)
cv2.imwrite(outputFileName,img)