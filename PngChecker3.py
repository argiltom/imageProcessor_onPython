import sys
#print(sys.argv[0])#どうやら第零引数にはソースファイルの名前が格納されるみたい
import numpy as np
import cv2
inputFileName1=sys.argv[1]
inputFileName2=sys.argv[2]
isSame=1
notsamePoint=[]
img1=np.float64(cv2.imread(inputFileName1))
img2=np.float64(cv2.imread(inputFileName2))

if(img1.shape[0]!=img2.shape[0] or img1.shape[1]!=img2.shape[1]):
    isSame=0
else:
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            for k in range(3):
                if(img1[i,j,k]!=img2[i,j,k]):
                    isSame=0
                    notsamePoint.append((i,j))
                    print("合わなかった画素の比較",img1[i,j,k],img2[i,j,k])

if(isSame==1):
    print("二つの画像の画素は等しいです")
elif(isSame==0):
    print("等しくない画像です")
    print("等しくない地点:"+str(len(notsamePoint)))
    img1=np.float64(cv2.imread(inputFileName1))
    for i in range(len(notsamePoint)):
        img1[notsamePoint[i]]=(0,0,255)
    cv2.imshow("diff",img1)
    cv2.waitKey(0)

