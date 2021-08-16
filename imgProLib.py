#全てのことが自分にとって可能であるという前提に立つ！ そうすれば、「どうすればできるか？」を考え始めることが出来るようになる
#考え始めさえすれば、全ての課題は解決可能になる！　　
#諦めなければ希望はある！、　炎は此処に！

import numpy as np
import sys
import cv2
import copy
import math


class imgProCls:
    def __init__(self,img):
        self.queue=[]#キュー
        self.img=img
        pass
    def EnQuere(self,y,x):
        self.queue.append((y,x))
    def DeQuere(self):
        if len(self.queue)==0:
            return -1,-1
        (y,x)=self.queue[0]
        del self.queue[0]
        return y,x
    #まずは大津法！
    def calcPixnumMeanVari(self,histo,value):
        num  = np.sum(histo)
        if num==0:
            return 0,0,0
        mean = np.sum(histo * value) / num
        vari = np.sum(histo * ( (value - mean)**2) ) / num
        return num, mean, vari


    #大津法　標準出力にエラーが出ると困るので、修正
    def OutuMethod(self):

        #画像を引数にとり輝度画像へ変換
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)


        H,W=img.shape[0],img.shape[1]
        #---------------------------------------

        #ここを編集

        #TODO 1 ヒストグラムを計算
        histo =np.zeros((256),int)
        for y in range(H):
            for x in range(W):
                histo[img[y,x]]+=1

        # 2 ヒストグラムから全体の画素数・平均・分散を計算
        value =  np.arange(256)

        num, mean, vari = self.calcPixnumMeanVari(histo, value) #よくわからんが、動くからヨシ！


        #TODO 3 Otsu法により閾値threshを計算
        #ヒント: スライスをうまく使うとコードが綺麗にかけます
        #calcPixnumMeanVari(histo[a:b], value[a:b])　
        #上のようにとすると [a,b)の範囲の平均値、分散値を計算できます．
        #最大分離度
        maxDegreeV=0
        thresh = 0
        for i in range(1,255):
            #Arrey[0:10]とやると,0～9の要素のみを引っ張り出すことが出来る
            #histo[0:i]とやると0～i-1までの要素を引っ張ってこれる
            #histo[i:max]とやるとi～maxまでの要素を引っ張ってこれる
            numB,meanB,variB = self.calcPixnumMeanVari(histo[0:i],value[0:i])
            numW,meanW,variW = self.calcPixnumMeanVari(histo[i:256],value[i:256])
            #クラス内分散　小さいほうを取る
            interVari=(numB*variB+numW*variW)/(numB+numW)
            #クラス間分散 大きいほうを取る
            betweenVari=(numB*(meanB-mean)**2+numW*(meanW-mean)**2)/(numB+numW)
            #分離度
            degreeV=betweenVari/interVari
            if maxDegreeV<degreeV:
                    maxDegreeV=degreeV
                    thresh=i #閾値を更新 
        #画像の二値化（boolean array indexingによる実装）
        img[img >= thresh] = 255
        img[img <  thresh] = 0
        return img
    #---------------------------------------
    #大津法終わり！
    #円形のモーフォロジー演算を行う
    #option=0=Dilation(膨張) option=1=Eilation(収縮)
    def Morphology(self,r,option):
        #カーネルの作成
        kernel=np.zeros((2*r+1,2*r+1),dtype=int)
        #円形カーネルの設定
        for y in range(-r,r+1):
            for x in range(-r,r+1):
                if(r**2>=y**2+x**2):
                    kernel[r+y,r+x]=1
                
        #print(kernel) #r=2で5×5の丁度良いカーネル
        
        #カーネルに合わせて膨張処理
        H,W=self.img.shape[0],self.img.shape[1]
        retImg = np.zeros_like(self.img)
        for y in range(r,H-r):
            for x in range(r,W-r):
                #前景領域
                    #まず、カーネルと画像のスライスのアダマール積を取り、生き残った画素の最大値を取る
                    outKernel=self.img[(y-r):(y+r+1),(x-r):(x+r+1)]*kernel
                    targetList=[]
                    for i in range(outKernel.shape[0]):
                        for j in range(outKernel.shape[1]):
                            if(kernel[i,j]==1):
                                targetList.append(outKernel[i,j])
                    if(option==0):#膨張
                        #print("max="+str(np.max(targetList)))
                        retImg[y,x]=np.max(targetList)
                    if(option==1):#縮小
                        #print("min="+str(np.max(targetList)))
                        retImg[y,x]=np.min(targetList)
                    #print(temp)
                    #print(str(tempMax))
                    
        return retImg

    #RGB対応のモーフォロジー R+G+Bでの最大値最小値で、RGBで一組としてモーフォロジーを行う
    def MorphologyRGB(self,r,option):#thresh前景領域とみなす閾値　#thresh<=の画素に対してモーフォロジーを適用
        
        def GetListMinMax(list):
            minV,minPix=list[0]
            maxV,maxPix=list[0]
            for i in range(len(list)):
                tempV,tempPix=list[i]
                if minV > tempV:
                    minV=tempV
                    minPix=tempPix
                if maxV < tempV:
                    maxV = tempV
                    maxPix = tempPix
            return minPix,maxPix
        
        #カーネルの作成
        kernel=np.zeros((2*r+1,2*r+1),dtype=int)
        #円形カーネルの設定
        for y in range(-r,r+1):
            for x in range(-r,r+1):
                if(r**2>=y**2+x**2):
                    kernel[r+y,r+x]=1
                
        #print(kernel) #r=2で5×5の丁度良いカーネル
        
        #カーネルに合わせて膨張処理
        H,W=self.img.shape[0],self.img.shape[1]
        retImg = np.zeros_like(self.img)
        
        for y in range(r,H-r):
            for x in range(r,W-r):
                
                #まず、カーネルと画像のスライスのアダマール積を取り、生き残った画素の最大値を取る
                outKernel=self.img[(y-r):(y+r+1),(x-r):(x+r+1),0]*kernel
                outKernel=outKernel+self.img[(y-r):(y+r+1),(x-r):(x+r+1),1]*kernel
                outKernel=outKernel+self.img[(y-r):(y+r+1),(x-r):(x+r+1),2]*kernel
                targetList=[]
                for i in range(outKernel.shape[0]):
                    for j in range(outKernel.shape[1]):
                        if(kernel[i,j]==1):
                            targetList.append((outKernel[i,j],self.img[y-r+i,x-r+j]))
                minPix,maxPix = GetListMinMax(targetList)
                if(option==0):#膨張
                    #print("max="+str(np.max(targetList)))
                    retImg[y,x]=maxPix
                if(option==1):#縮小
                    #print("min="+str(np.max(targetList)))
                    retImg[y,x]=minPix    
                    #print(temp)
                    #print(str(tempMax))
                    
        return retImg

    #RGB対応のモーフォロジー RGBをそれぞれ独立させて、最大最小画素をRGBごとに取りモーフォロジー演算を行う
    def MorphologyRGB2(self,r,option):#thresh前景領域とみなす閾値　#thresh<=の画素に対してモーフォロジーを適用
        #カーネルの作成
        kernel=np.zeros((2*r+1,2*r+1),dtype=int)
        #円形カーネルの設定
        for y in range(-r,r+1):
            for x in range(-r,r+1):
                if(r**2>=y**2+x**2):
                    kernel[r+y,r+x]=1
                
        #print(kernel) #r=2で5×5の丁度良いカーネル
        
        #カーネルに合わせて膨張処理
        H,W=self.img.shape[0],self.img.shape[1]
        retImg = np.zeros_like(self.img)
        for y in range(r,H-r):
            for x in range(r,W-r):
                for k in range(3):
                #前景領域
                    #まず、カーネルと画像のスライスのアダマール積を取り、生き残った画素の最大値を取る
                    outKernel=self.img[(y-r):(y+r+1),(x-r):(x+r+1),k]*kernel
                    targetList=[]
                    for i in range(outKernel.shape[0]):
                        for j in range(outKernel.shape[1]):
                            if(kernel[i,j]==1):
                                targetList.append(outKernel[i,j])
                    if(option==0):#膨張
                        #print("max="+str(np.max(targetList)))
                        retImg[y,x,k]=np.max(targetList)
                    if(option==1):#縮小
                        #print("min="+str(np.max(targetList)))
                        retImg[y,x,k]=np.min(targetList)
                    #print(temp)
                    #print(str(tempMax))
                    
        return retImg


    #シード画素を領域成長させて、該当領域の座標リストを返す
    #RGB画像に対応した領域成長
    #allowRangeは,rgbそれぞれのシード画素からの許容するプラスマイナスの差異を定義する．
    #格納されているimgが白黒画像の時は、IndexError: invalid index to scalar variable.を起こしてしまうので注意
    def GrowthPoint(self,seed_y,seed_x,allowRange):
        #スタート地点の画素を取得
        startPoint=self.img[seed_y,seed_x]
        bin_img = np.zeros((self.img.shape[0],self.img.shape[1]))
        bin_img[0:bin_img.shape[0],0:bin_img.shape[1]] = 1 #全部1で初期化 1は未踏破の証
        bin_img[seed_y,seed_x]=255 #seed画素が前景であることは確定
        self.EnQuere(seed_y,seed_x)
        #返す座標リスト
        retList=[]
        while len(self.queue)!=0:
            y,x=self.DeQuere()
            #print(str(y)+","+str(x)+" thresh="+str(allowRange))
            #隣接画素を取得
            for i in range(-1,2):
                if bin_img[y+i,x]==1:
                    isEnquere=True
                    for k in range(3):
                        if (startPoint[k]-allowRange)<=self.img[y+i,x,k] and self.img[y+i,x,k]<=(startPoint[k]+allowRange):
                            pass
                        else:
                            isEnquere=False
                    if isEnquere :
                        bin_img[y+i,x]=255
                        self.EnQuere(y+i,x)
                        retList.append((y+i,x))#同じ領域として追加
                    else:
                        bin_img[y+i,x]=0
            for i in range(-1,2):
                if bin_img[y,x+i]==1:
                    isEnquere=True
                    for k in range(3):
                        if (startPoint[k]-allowRange)<=self.img[y,x+i,k] and self.img[y,x+i,k]<=(startPoint[k]+allowRange):
                            pass
                        else:
                            isEnquere=False
                            
                    if isEnquere :
                        bin_img[y,x+i]=255
                        self.EnQuere(y,x+i)
                        retList.append((y,x+i))#同じ領域として追加
                    else:
                        bin_img[y,x+i]=0
            
        return retList

    #地点を指定、±画素閾値範囲を指定して、色を指定する →　その色で領域を塗りつぶした画像を取得する
    def GrowthPointPainter(self,seed_y,seed_x,allowRange,color:list):
        paintList=self.GrowthPoint(seed_y,seed_x,allowRange)
        retImg=copy.deepcopy(self.img)
        for y,x in paintList:
            retImg[y,x,0]=color[2]
            retImg[y,x,1]=color[1]
            retImg[y,x,2]=color[0]
            pass
        return retImg
    
    def ErrorDiffusionHalfTone(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        img = np.float64(img)
        #出力画像を準備
        H = img.shape[0]
        W = img.shape[1]
        img_out = np.zeros((H,W), np.uint8)

        #誤差拡散法の計算
        for y in range(H)  :
            for x in range(W)  :
                
                if(img[y,x]>127):
                    gosa=img[y,x]-255
                    img_out[y,x]=255
                else:
                    gosa=img[y,x]-0
                    img_out[y,x]=0
                #誤差拡散
                if(x<W-1):
                    img[y,x+1]+=gosa*5/16
                if(y<H-1):
                    img[y+1,x]+=gosa*5/16
                    if(x>0):
                        img[y+1,x-1]+=gosa*3/16
                    if(x<W-1):
                        img[y+1,x+1]+=gosa*3/16
        return img_out
    #CV2のキャニーフィルタをそのまま使っている
    def Canny(self,threshold1,threshold2):
        #http://opencv.jp/opencv-2svn/cpp/feature_detection.html
        #img=cv2.Canny(self.img,threshold2=170,threshold1=90,L2gradient=True)
        img=cv2.Canny(self.img,threshold1,threshold2,L2gradient=True)
        return img
    #線形フィルタの演算装置 #RGBの画像を前提としている
    def LinearFilter(self,filter:np.ndarray):
        #フィルタの例
        #filter=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.uint8)#ndarrayインスタンスを作成
        #ガウシアンフィルタ
        #filter=np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=np.uint8)#ndarrayインスタンスを作成
        #filter=filter/16
        #画像の初期化はこのようにやる--------------------------------------
        tempImg=np.float64(self.img)
        retImg=np.zeros_like(self.img,np.uint8)
        #----------------------------------------------------------------
        fh=filter.shape[0]
        fw=filter.shape[1]
        H,W=self.img.shape[0],self.img.shape[1]
        
        
        for y in range(0+math.floor(fh/2),H-math.floor(fh/2)):
            for x in range(0+math.floor(fw/2),W-math.floor(fw/2)):
                temp=0
                for fy in range(fh):
                    for fx in range(fw):
                        temp+=tempImg[y-math.floor(fh/2)+fy,x-math.floor(fw/2)+fx]*filter[fy,fx]#これで三画素文纏めて計算できる
                        #print(temp)
                retImg[y,x]=temp
        return retImg
    #ガウシアンフィルタを生成し、LinearFilterでフィルタを適用し、画像を出力
    def GaussianFilter(self,length,ρ):
        def CalcGaussian(y,x,ρ):
            return 1/(2*math.pi*(ρ**2))*math.exp(-(x**2+y**2)/(2*(ρ**2)))
            
        gaussF=np.zeros((length,length))
        for y in range(length):
            for x in range(length):
                gaussF[y,x]=CalcGaussian(y-math.floor(length/2),x-math.floor(length/2),ρ)
                print(y-math.floor(length/2),x-math.floor(length/2),end="   ")
            print("")
        
        #https://imagingsolution.net/imaging/gaussian/ ガウシアンフィルタについての参考資料

        gaussF*=1/sum(sum(gaussF))#正規化#ガウシアンフィルタの総和が1になるように調整
        print(gaussF)
        print(sum(sum(gaussF)))#ガウシアンフィルタのフィルタ総和を出力

        return self.LinearFilter(gaussF)
        
    #privateMethod
    def __swapRGB(self,index1:int,index2:int):
        retImg=copy.deepcopy(self.img)
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                retImg[y,x,index1]=self.img[y,x,index2]
                retImg[y,x,index2]=self.img[y,x,index1]
        return  retImg
        
    def swapRG(self):
        return self.__swapRGB(1,2)
    def swapRB(self):
        return self.__swapRGB(0,2)
    def swapGB(self):
        return self.__swapRGB(0,1)
        
        

#---------------------------------------
#メイン処理
#load image
#fname_in  = sys.argv[1]
#img = cv2.imread(fname_in)

#cv2.imshow("img",img)
#cv2.waitKey(0)
#勾配情報出力
#img=cv2.Canny(img,threshold2=170,threshold1=90,L2gradient=True)
