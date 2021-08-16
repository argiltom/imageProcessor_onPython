import sys
import numpy as np
import math
import cv2
import imgProLib

fname_in  = sys.argv[1]
img = cv2.imread(fname_in)
#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
#pro.img=pro.GrowthPointPainter(0,0,100,(255,0,0))
#retImg=pro.AbsDiffImg(img)
#cv2.imshow("img",retImg)
#cv2.waitKey(0)

#filter=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])#ndarrayインスタンスを作成
#print(filter)
#retImg=pro.LinearFilter(filter)
#cv2.imshow("img",retImg)
#cv2.waitKey(0)

#filter=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])#ndarrayインスタンスを作成
#filter=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])#ndarrayインスタンスを作成
#print(filter)
#retImg=pro.LinearFilter(filter)
#cv2.imshow("img",retImg)
#cv2.waitKey(0)
#filter=np.array([[1,2,1],[2,4,2],[1,2,1]])#ndarrayインスタンスを作成
#filter=filter/16
#print(filter)
#gauss1=pro.GaussianFilter(3,0.8)
#gauss2=pro.GaussianFilter(5,1)
#Dog=gauss2-gauss1
#cv2.imshow("img",Dog)
#cv2.waitKey(0)

#retImg=imgProLib.imgProCls(imgProLib.imgProCls(img).swapRB()).swapRB()
#cv2.imshow("img",retImg)
#cv2.waitKey(0)
# retImg=pro.GrowthPointPainter(15,15,60,(222,111,32))
# cv2.imshow("img",retImg)
# cv2.waitKey(0)

#retImg=pro.CannyGaussian(100,200,5,5,1)

#retImg=pro.MorphologyRGB(1,1)
retImg=pro.SwapRB()


cv2.imshow("img",retImg)
cv2.waitKey(0)
#retImg=pro.GaussianFilter(15,5)
#cv2.imshow("img",retImg)
#cv2.waitKey(0)
print("OutPutFileName=",end="")
outputStr=input()
cv2.imwrite(outputStr,retImg)

