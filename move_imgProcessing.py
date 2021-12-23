#import
import cv2
import numpy as np
import mediapipe as mp
#静止画表示に使用
import matplotlib.pyplot as plt
import sys
#一通り動画を再生
import time


def save_video( frame_W, frame_H,imgList,outputFileName ) :

    #cap.set( cv2.CAP_PROP_POS_FRAMES, 0 )
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out    = cv2.VideoWriter( outputFileName , fourcc, 20.0, (frame_W, frame_H) )
    for i in range(len(imgList)):
        out.write( imgList[i] )
    out.release()
#ReadVidio
fileName=sys.argv[1]
cap = cv2.VideoCapture(fileName)

#img一覧
imgList=[]





#最初の一フレームだけ読み込む
ret,frame=cap.read()
print("動画の読み込み成否:"+str(ret))

fcount=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while cap.isOpened():#再生できるなら、次のを取り出して進む
    #cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    
    ret,img=cap.read()
    
    if ret :
        #ここに画像処理をして、それをAppendしてやればいい
        #cv2.rectangle(img,(0+centerX,0+centerY),(x_size+centerX,y_size+centerY),(255,0,0),1)
        imgList.append(img)
        #print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        #cv2.imshow("temp",img)
        #key = cv2.waitKey(0)
        ##time.sleep(1)
    else:
        break
cap.release()

save_video(imgList[0].shape[1],imgList[0].shape[0],imgList,"output.mp4")