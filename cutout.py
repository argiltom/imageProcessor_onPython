import sys
import numpy as np
import math
import cv2
import time
from imgProLib import imgProCls

def MouseEvent(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        print(y,x)
        pro.img=pro.GrowthPointAlphaPainter(y,x,30,0)
        #pro.img[y,x,:]=255

fname_in  = sys.argv[1]
img = cv2.imread(fname_in)
pro=imgProCls(img)


filter=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])#ndarrayインスタンスを作成
#ウィンドウ生成
cv2.namedWindow("ImgWindow",cv2.WINDOW_KEEPRATIO)

#マウスイベントセット
cv2.setMouseCallback("ImgWindow",MouseEvent)

while True:
    pro.img=pro.LinearFilter(filter)
    cv2.imshow("ImgWindow",pro.AlphaImg2RGBImg((255,255,255)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

print("OutPutFileName=",end="")
outputStr=input()
cv2.imwrite(outputStr,pro.img)

