import sys
import numpy as np
import math
import cv2
import time
from imgProLib import imgProCls
fname_in  = sys.argv[1]
r = int(sys.argv[2])
img = cv2.imread(fname_in)
pro=imgProCls(img)

t1_1=time.time()
retImg=pro.MedianFilter(r)
t1_2=time.time()
print(str(t1_2-t1_1)+"秒")
cv2.imshow("img",retImg)
cv2.waitKey(0)



print("OutPutFileName=",end="")
outputStr=input()
cv2.imwrite(outputStr,retImg)

