import sys
import numpy as np
import math
import cv2
import time
from imgProLib import imgProCls
fname_in  = sys.argv[1]
img = cv2.imread(fname_in)
pro=imgProCls(img)
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

# t1_1=time.time()
# retImg1=pro.Canny(100,200)
# t1_2=time.time()
# print(str(t1_2-t1_1)+"秒")
# cv2.imshow("img",retImg1)
# cv2.waitKey(0)


# t1_1=time.time()
# retImg1=cv2.GaussianBlur(pro.img,(5,5),100)
# t1_2=time.time()
# print(str(t1_2-t1_1)+"秒")

# cv2.imshow("img",retImg1)
# cv2.waitKey(0)
filter=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])#ndarrayインスタンスを作成
#filter=np.array([[0,1,0],[1,-4,1],[0,1,0]])#ラプラシアンフィルタ
t1_1=time.time()
#retImg=pro.LinearFilter(filter)
#retImg=pro.YABAIFilter(10)
#retImg=pro.SUGOIFilter(5)
#retImg=pro.Canny(100,200)
#retImg=pro.SwapGB()
pro.img=pro.GrowthPointAlphaPainter(10,10,140,0)
#pro.img=retImg
pro.img=pro.MorphologyRGB(3,0)
#pro.img=pro.GaussianFilter(10,5)
t1_2=time.time()
print(str(t1_2-t1_1)+"秒")

cv2.imshow("img",pro.AlphaImg2RGBImg((255,255,255)))
cv2.waitKey(0)


#print("OutPutFileName=",end="")
#outputStr=input()
#cv2.imwrite(outputStr,retImg)

# t1_1=time.time()
# retImg=pro.SwapGB()
# t1_2=time.time()
# print(str(t1_2-t1_1)+"秒")

# cv2.imshow("img",retImg)
# cv2.waitKey(0)


print("OutPutFileName=",end="")
outputStr=input()
cv2.imwrite(outputStr,pro.img)

