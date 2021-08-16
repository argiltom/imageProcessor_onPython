import sys
import numpy as np
import math
import cv2
import imgProLib

fname_in  = sys.argv[1]
img = cv2.imread(fname_in)
pro=imgProLib.imgProCls(img)
# retImg=pro.MorphologyRGB(2,0)
# cv2.imshow("img",retImg)
# cv2.waitKey(0)
# retImg=pro.MorphologyRGB(2,1)
# cv2.imshow("img",retImg)
# cv2.waitKey(0)
# retImg=pro.ErrorDiffusionHalfTone()
# cv2.imshow("img",retImg)
# cv2.waitKey(0)
#retImg=pro.Canny(100,200)
#pro=imgProLib.imgProCls(retImg)
#retImg=pro.GrowthPointPainter(0,0,100,(255,255,0))

#filter=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])#ndarrayインスタンスを作成
#print(filter)
#retImg=pro.LinearFilter(filter)
#cv2.imshow("img",retImg)
#cv2.waitKey(0)

#filter=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])#ndarrayインスタンスを作成
#print(filter)

#filter=np.array([[1,2,1],[2,4,2],[1,2,1]])#ndarrayインスタンスを作成
#filter=filter/16
#print(filter)
retImg=pro.GaussianFilter(3,20)
cv2.imshow("img",retImg)
cv2.waitKey(0)

retImg=imgProLib.imgProCls(imgProLib.imgProCls(img).swapRB()).swapRB()
cv2.imshow("img",retImg)
cv2.waitKey(0)

print("OutPutFileName=",end="")
outputStr=input()
cv2.imwrite(outputStr,retImg)

