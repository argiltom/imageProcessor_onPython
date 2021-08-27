#全てのことが自分にとって可能であるという前提に立つ！ そうすれば、「どうすればできるか？」を考え始めることが出来るようになる
#考え始めさえすれば、全ての課題は解決可能になる！　　
#諦めなければ希望はある！、　炎は此処に！
#img.ndim == 3 →　RGB
#img.ndim == 2 →　GRAY

import numpy as np
import sys
import cv2
import copy
import math
import random
from numpy.core.fromnumeric import shape
from numpy.core.numeric import zeros_like
#from PIL import Image

#https://atsuoishimoto.hatenablog.com/entry/2018/01/06/195649 pythonの処理が遅い原因について
#for文自体が遅いわけではない、
#Pythonの演算が遅い最大の要因は、Pythonが静的な型宣言を行わない言語で、型推論もJITもなく、常に動的にオブジェクトの演算を行う、という点にある場合がほとんどだ。(引用)

class imgProCls:
    def __init__(self,img):
        self.queue=[]#キュー
        self.img=img
        pass
    def __EnQuere(self,y,x):
        self.queue.append((y,x))
    def __DeQuere(self):
        if len(self.queue)==0:
            return -1,-1
        (y,x)=self.queue[0]
        del self.queue[0]
        return y,x
    #imgに格納されている

    #self.imgをRGBA方式に変更する  そしてRGBA方式を共通規格とする．
    def __SelfImgConvert2RGBA(self):
        if self.img.ndim == 2:
            self.img = cv2.cvtColor(self.img,cv2.COLOR_GRAY2RGB)
        #カラー画像を透過考慮画像RGBA画像に変更
        if self.img.shape[2]==3: #alphaがないなら
            self.img=cv2.cvtColor(self.img,cv2.COLOR_RGB2RGBA)

    
    #画像を演算用画像に書き換える

    #画像を出力用画像に書き換える．


    #透過画像がimgに格納されているなら、それをRGB方式にして出力する JPG出力用に変換するときなどにどうぞ
    def AlphaImg2RGBImg(self,backColor:list):
        retImg=np.zeros_like(self.img)
        self.__SelfImgConvert2RGBA()

        for y in range(retImg.shape[0]):
            for x in range(retImg.shape[1]):
                alpha=self.img[y,x,3]/255
                point=self.img[y,x]
                retImg[y,x,0]=(1-alpha)*backColor[2]+alpha*point[0]
                retImg[y,x,1]=(1-alpha)*backColor[1]+alpha*point[1]
                retImg[y,x,2]=(1-alpha)*backColor[0]+alpha*point[2]


        return retImg

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
    #RGB画像,GRAY画像両方に対応した領域成長
    #allowRangeは,rgbそれぞれのシード画素からの許容するプラスマイナスの差異を定義する．
    #格納されているimgが白黒画像の時は、IndexError: invalid index to scalar variable.を起こしてしまうので注意→修正完了白黒画像でもちゃんと動くようになった．
    def GrowthPoint(self,seed_y,seed_x,allowRange):
        self.__SelfImgConvert2RGBA()#これでエラーを防ぐ
        #スタート地点の画素を取得
        startPoint=self.img[seed_y,seed_x]
        #領域演算画像
        H,W=self.img.shape[0],self.img.shape[1]
        bin_img = np.zeros((self.img.shape[0],self.img.shape[1]))
        bin_img[0:bin_img.shape[0],0:bin_img.shape[1]] = 1 #全部1で初期化 1は未踏破の証
        bin_img[seed_y,seed_x]=255 #seed画素が前景であることは確定
        self.__EnQuere(seed_y,seed_x)
        
        #返す座標リスト
        retList=[]
        while len(self.queue)!=0:
            y,x=self.__DeQuere()
            #print(str(y)+","+str(x)+" thresh="+str(allowRange))
            #隣接画素を取得
            for i in range(-1,2):
                if (y+i)<0 or (y+i)>=H: #踏み越え防止
                    continue
                if bin_img[y+i,x]==1:
                    isEnquere=True
                    for k in range(3):
                        if (startPoint[k]-allowRange)<=self.img[y+i,x,k] and self.img[y+i,x,k]<=(startPoint[k]+allowRange):
                            pass
                        else:
                            isEnquere=False
                    if isEnquere :
                        bin_img[y+i,x]=255
                        self.__EnQuere(y+i,x)
                        retList.append((y+i,x))#同じ領域として追加
                    else:
                        bin_img[y+i,x]=0
            for i in range(-1,2):
                if (x+i)<0 or (x+i)>=W: #踏み越え防止
                    continue
                if bin_img[y,x+i]==1:
                    isEnquere=True
                    for k in range(3):
                        if (startPoint[k]-allowRange)<=self.img[y,x+i,k] and self.img[y,x+i,k]<=(startPoint[k]+allowRange):
                            pass
                        else:
                            isEnquere=False
                            
                    if isEnquere :
                        bin_img[y,x+i]=255
                        self.__EnQuere(y,x+i)
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
    #地点を指定、±画素閾値範囲を指定して、α値を指定する そのα値で塗りつぶした画像を取得する これは透過画像として出力されるため、この画像を再代入は出来ない
    def GrowthPointAlphaPainter(self,seed_y,seed_x,allowRange,alphaVal:np.uint8):
        paintList=self.GrowthPoint(seed_y,seed_x,allowRange)
        retImg=copy.deepcopy(self.img)
        if retImg.shape[2]==3:
            retImg=cv2.cvtColor(retImg,cv2.COLOR_RGB2RGBA)
            #print(str("cv2.cvtColor(retImg,cv2.COLOR_RGB2BGRA).ndim=")+str(retImg.ndim))
        for y,x in paintList:
            retImg[y,x,3]=alphaVal
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

    #--------------------------------------------------------------------------------------フィルタ-----------------------------------------------------------
    #線形フィルタの演算装置 ついに透過画像にも対応した.
    def LinearFilter(self,filter:np.ndarray):
        #フィルタの例
        #filter=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.uint8)#ndarrayインスタンスを作成
        self.__SelfImgConvert2RGBA()#画像をRGBAに変換する
        #画像の初期化はこのようにやる--------------------------------------
        tempImg=np.float64(self.img)
        retImg=np.zeros_like(self.img,np.uint8)
        #----------------------------------------------------------------
        fh=filter.shape[0]
        fw=filter.shape[1]
        H,W=self.img.shape[0],self.img.shape[1]
        fh_2f=math.floor(fh/2)
        fw_2f=math.floor(fw/2)
        fh_2c=math.ceil(fh/2)
        fw_2c=math.ceil(fw/2)
        print(filter[0:fh,0:fw])
        for y in range(0+fh_2f,H-fh_2f):
            for x in range(0+fw_2f,W-fw_2f):
                for k in range(4):
                    temp=0
                    temp+=sum(sum(tempImg[y-fh_2f:y+fh_2c,x-fw_2f:x+fw_2c,k]*filter[0:fh,0:fw]))
                        #print(temp)
                #値の調整(255>x)→x=255 (x<0)→x=0
                    if temp>255:
                        temp=255
                    elif temp<0:
                        temp=0
                    retImg[y,x,k]=temp
        return retImg
    #ガウシアンフィルタを生成し、LinearFilterでフィルタを適用し、画像を出力
    def GaussianFilter(self,length,ρ):
        def CalcGaussian(y,x,ρ):
            return 1/(2*math.pi*(ρ**2))*math.exp(-(x**2+y**2)/(2*(ρ**2)))
        #ガウシアンフィルタの生成
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
    def __SwapRGB(self,index1:int,index2:int):
        retImg=copy.deepcopy(self.img)
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                retImg[y,x,index1]=self.img[y,x,index2]
                retImg[y,x,index2]=self.img[y,x,index1]
        return  retImg
        
    def SwapRG(self):
        return self.__SwapRGB(1,2)
    def SwapRB(self):
        return self.__SwapRGB(0,2)
    def SwapGB(self):
        return self.__SwapRGB(0,1)
    #クラスが保有している画像から,引数の画像を引いた画像を出力する isAbsがTrueなら差の絶対値を取る
    def DiffImg(self,img:np.ndarray,isAbs:bool):
        if img.ndim == 2:       #グレースケール画像をRGB画像に変換　これでエラーを未然に防ぐ
            img= cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if self.img.ndim == 2:      #グレースケール画像をRGB画像に変換　これでエラーを未然に防ぐ
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        #出力画像の初期化
        retImg=np.zeros_like(self.img,np.uint8)
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                temp=self.img[y,x]-img[y,x]
                for k in range(3):
                    if temp[k]<0:
                        if isAbs:
                            temp[k]=-temp[k]
                        else:
                            temp[k]=0
                retImg[y,x]=temp
        return retImg
    #CannyFilterで検出した箇所にガウシアンフィルタを掛ける
    #輪郭部を効率的に平滑化することを目的としている． #でもガウシアンフィルタの処理より遅い
    def CannyGaussian(self,threshold1,threshold2,guaussianLen,ρ,round):
        def CalcGaussian(y,x,ρ):
            return 1/(2*math.pi*(ρ**2))*math.exp(-(x**2+y**2)/(2*(ρ**2)))
        
        cannyImg=cv2.Canny(self.img,threshold1,threshold2,L2gradient=True)
        cannylist=[]
        for y in range(cannyImg.shape[0]):
            for x in range(cannyImg.shape[1]):
                if(cannyImg[y,x]==255):
                    cannylist.append((y,x))
        
        #roundで指定したcanny検出周囲の画素にもガウシアンフィルタを掛ける
        #まず,元のcannylistのコピーを取る
        copyCannyList=copy.deepcopy(cannylist)
        #円形で設定 元のリストの周囲のも追加する
        for y,x in copyCannyList:
            for ry in range(-round,round+1):
                for rx in range(-round,round+1):
                    if(round**2>=ry**2+rx**2):

                        if not (y+ry,x+rx) in cannylist: #重ねて追加しないようにする
                            cannylist.append((y+ry,x+rx))
        #テスト
        #print(cannylist)
        #ガウシアンフィルタの定義
        gaussF=np.zeros((guaussianLen,guaussianLen))
        for y in range(guaussianLen):
            for x in range(guaussianLen):
                gaussF[y,x]=CalcGaussian(y-math.floor(guaussianLen/2),x-math.floor(guaussianLen/2),ρ)
        gaussF*=1/sum(sum(gaussF))#正規化#ガウシアンフィルタの総和が1になるように調整
        gfh,gfw=guaussianLen,guaussianLen
        
        #画像の初期化
        tempImg=np.float64(self.img)
        
        retImg=copy.deepcopy(self.img)
        #retImg=np.zeros_like(self.img) #テスト用 これをONにすると、ガウシアンした箇所だけ表示できるようになる
        #画像演算
        for y,x in cannylist:
            #print(y,x)
            y_left=math.floor(gfh/2)
            x_left=math.floor(gfw/2)
            y_right=tempImg.shape[0]-math.ceil(gfh/2)
            x_right=tempImg.shape[1]-math.ceil(gfw/2)
            temp=0
            #配列踏み越え防止
            if y<(y_left) or y>(y_right):#floorだとたまに踏み越えが起こる
                #guprint(y,x)
                pass
            elif x<(x_left) or x>(x_right):
                #print(y,x)
                pass
            else:
                for fy in range(gfh):
                    for fx in range(gfw):
                        temp+=tempImg[y-y_left+fy,x-x_left+fx]*gaussF[fy,fx]
                retImg[y,x]=temp
        return retImg


    
    #すごいフィルタ
    def SUGOIFilter(self,flen:int):
        filter=np.zeros((flen,flen))
        for y in range(flen):
            for x in range(flen):
                filter[y,x]=random.randint(-100,100)
        filter=filter/sum(sum(filter))#総和が1になる様に正規化
        #print(filter)
        return self.LinearFilter(filter)
    #YABAIフィルタ (和名:荒らしフィルタ)
    def YABAIFilter(self,n:int):
        tempPro=imgProCls(self.img)
        doneList=[]
        for i in range(n):
            num=random.randint(0,7)
            
            if num==0:
                doneList.append("精鋭化")
                #精鋭化フィルタ
                filter=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                tempPro.img=tempPro.LinearFilter(filter)
            elif num==1:
                doneList.append("Gaussian")
                tempPro.img=tempPro.GaussianFilter(3,0.8)
            elif num==2:
                doneList.append("膨張")
                tempPro.img=tempPro.MorphologyRGB(2,0)
            elif num==3:
                doneList.append("収縮")
                tempPro.img=tempPro.MorphologyRGB(2,1)
            elif num==4:
                doneList.append("すごいフィルタ")
                tempPro.img=tempPro.SUGOIFilter(3)
            elif num==5:
                doneList.append("R←→G")
                tempPro.img=tempPro.SwapRG()
            elif num==6:
                doneList.append("R←→B")
                tempPro.img=tempPro.SwapRB()
            elif num==7:
                doneList.append("G←→B")
                tempPro.img=tempPro.SwapGB()
        print(doneList)
        return tempPro.img
    #--------------------------------------------------------------------------------------/フィルタ-----------------------------------------------------------
#---------------------------------------
    #かつての遺物 スライスによってより高速に改良されたため没
    # def LinearFilterOld(self,filter:np.ndarray):
    #     #フィルタの例
    #     #filter=np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.uint8)#ndarrayインスタンスを作成
    #     #ガウシアンフィルタ
    #     #filter=np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=np.uint8)#ndarrayインスタンスを作成
    #     #filter=filter/16
    #     #画像の初期化はこのようにやる--------------------------------------
    #     tempImg=np.float64(self.img)
    #     retImg=np.zeros_like(self.img,np.uint8)
    #     #----------------------------------------------------------------
    #     fh=filter.shape[0]
    #     fw=filter.shape[1]
    #     H,W=self.img.shape[0],self.img.shape[1]
        
        
    #     for y in range(0+math.floor(fh/2),H-math.floor(fh/2)):
    #         for x in range(0+math.floor(fw/2),W-math.floor(fw/2)):
    #             temp=0
    #             for fy in range(fh):
    #                 for fx in range(fw):
    #                     temp+=tempImg[y-math.floor(fh/2)+fy,x-math.floor(fw/2)+fx]*filter[fy,fx]#これで三画素文纏めて計算できる
    #                     #print(temp)
    #             #値の調整(255>x)→x=255 (x<0)→x=0
    #             #RGB画像時
    #             if tempImg.ndim == 3:
    #                 for k in range(3):
    #                     if temp[k]>255:
    #                         temp[k]=255
    #                     elif temp[k]<0:
    #                         temp[k]=0
    #             #グレースケール画像時
    #             elif tempImg.ndim == 2:
    #                 if temp>255:
    #                         temp=0
    #                 elif temp<0:
    #                         temp=0
                
    #             retImg[y,x]=temp
    #     return retImg










#メイン処理
#load image
#fname_in  = sys.argv[1]
#img = cv2.imread(fname_in)

#cv2.imshow("img",img)
#cv2.waitKey(0)
#勾配情報出力
#img=cv2.Canny(img,threshold2=170,threshold1=90,L2gradient=True)


#https://qiita.com/derodero24/items/f22c22b22451609908ee RGBA RGB GRAYの変換参考 https://qiita.com/Kazuhito/items/ff4d24cd012e40653d0c