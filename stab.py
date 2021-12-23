import sys
import numpy as np
import math
import cv2
import imgProLib

fname_in  = sys.argv[1]
img = cv2.imread(fname_in)
pro=imgProLib.imgProCls(img)
retImg=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#retImg=pro.ErrorDiffusionHalfTone()
#ZenkeiList=pro.GrowthPoint(200,200,58)
#retImg=pro.img
#for y,x in ZenkeiList:
#    retImg[y,x,0]=255
#    retImg[y,x,1]=255
#    retImg[y,x,2]=255
cv2.imshow("img",retImg)
cv2.waitKey(0)
print("OutPutFileName=",end="")
outputStr=input()
cv2.imwrite(outputStr,retImg)

