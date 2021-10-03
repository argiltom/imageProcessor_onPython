import sys
import numpy as np
import math
import cv2
import time
from imgProLib import imgProCls
fname_in1  = sys.argv[1]
fname_in2  = sys.argv[2]
img1 = cv2.imread(fname_in1)
img2 = cv2.imread(fname_in2)
pro1=imgProCls(img1)

t1_1=time.time()
retImg=pro1.DiffImg(img2,True)
t1_2=time.time()
print(str(t1_2-t1_1)+"秒")
cv2.imshow("img",retImg)
cv2.waitKey(0)


t1_1=time.time()
retImg=pro1.MorphologyRGB(5,0)
t1_2=time.time()
print(str(t1_2-t1_1)+"秒")
cv2.imshow("img",retImg)
cv2.waitKey(0)



print("OutPutFileName=",end="")
outputStr=input()
cv2.imwrite(outputStr,retImg)

