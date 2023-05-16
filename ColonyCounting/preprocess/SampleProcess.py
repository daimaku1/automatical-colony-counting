#!/usr/bin/env python
# _*_coding:utf-8_*_

"""
@Time :    2022/4/29 9:32
@Author:  qiaos
@File: SampleProcess.py
@Software: PyCharm
"""
import os

"""
@本文件主要用于：
完成的主要思路：

#todo:1
#todo:2
#todo:3

"""

import numpy as np
import cv2 as cv

import logging
import pickle

LOG_FILENAME ='ColonyMarking.history'
LOG_FORMAT = "[%(asctime)s.%(msecs)03d] %(filename)s %(funcName)s %(thread)d %(message)s "
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATEFMT, filename=LOG_FILENAME)

des_path = r'E:\Desktop\Images 4.17\Images 4.17\PCA VP 35H'
out_path = r'E:\Desktop'
if not os.path.exists(out_path):
    os.mkdir(out_path)
lightness = 1024

class Ellipse():
    """
    #椭圆类,保存从cv.fitEllipse 返回的值
    ((x,y),(long,short),angle)
    其中x,y为椭圆中心点的位置，a,b为长短轴直径，angle为中心旋转角度
    """
    center=None
    long=None
    short=None
    angle=None
    def __init__(self,ellipse):
        self.center=np.array(ellipse[0])
        self.long = ellipse[1][1]
        self.short= ellipse[1][0]
        self.angle= ellipse[2]

    def enlarge(self,n):
        self.center[0]-=n
        self.center[1]-= n
        # self.short+= n*2
        # self.long += n*2

    @property
    def centerInt(self):
        return round(self.center[0]),round(self.center[1])
    @property
    def radiusInt(self):
        return round((self.short+self.long)/4)

    @property
    def ellipse(self):
        return self.center,(self.long,self.short),self.angle


class ColorSpace():
    _mean=None 
    _std=None
    def __init__(self,mean,stdev):
        self._mean=mean
        self._std=stdev
    def __str__(self):
        return str(self._mean)+"_"+str(self._std)
    @property
    def mean(self):
        return self._mean
    @property
    def std(self):
        return self._std


class ConnectedComponents():
    """
    int count找到的连通区域数量
    int32 [count，5] status连接的统计  数组
    float32 [count，2] centroid：每个联通区域的中心点位置
    int32 [cols,rows] labels np 数组存储每个像素所属于的联通区域信息。
    """
    DEBUG=True
    class Status():
        """
         count,5 int32数组
         列内容 x起始坐标, y终点坐标, h高度, w宽度, s像素个数，
        """
        STATUS_COLS_X = 0
        STATUS_COLS_Y = 1
        STATUS_COLS_H = 2
        STATUS_COLS_W = 3
        STATUS_COLS_S = 4

        def __init__(self, status):
            # status=np.matrix([])
            self._status = status
        @property
        def areas(self):
            return self._status[:,self.STATUS_COLS_S]

        def getStatus(self, i):
            x = self._status[i][self.STATUS_COLS_X]
            y = self._status[i][self.STATUS_COLS_Y]
            w = self._status[i][self.STATUS_COLS_W]
            h = self._status[i][self.STATUS_COLS_H]
            s = self._status[i][self.STATUS_COLS_S]
            return x, y, w, h, s
        @property
        def data(self):
            return self._status

    class Labels():
        """
        labels的格式为rows,cols int32矩阵，每个点上保存分类值
        """

        def __init__(self, labels):
            self._labels = labels
        @property
        def data(self):
            return self._labels

    class Centroids():
        CENTROIDS_COLS_X = 0
        CENTROIDS_COLS_Y = 1

        def __init__(self, centroids):
            self._centroids = centroids
        @property
        def data(self):
            return self._centroids
    count = None
    labels = None
    status = None
    centroids = None
    _areaIndex=None

    def __init__(self, count, labels, status, centroids):
        self.count = count
        self.labels = self.Labels(labels)
        self.status = self.Status(status)
        self.centroids = self.Centroids(centroids)
        self.sortByArea()

    def sortByArea(self):
        """
        只能从到大排序，所以要注意哦
        :return:
        """
        index=np.argsort(self.status.areas)
        self._areaIndex=index

    @property
    def areaIndex(self):
        """
        面积从小到大顺序的索引
        只能从到大排序，所以要注意哦
        :return:
        """
        if self._areaIndex is None:
            self.sortByArea()
        return self._areaIndex

class Parts():
    """
    保存在status中的部件位置，
    >0具体位置
    -1没有设置
    -2没有找到
    -3程序出错
    """
    def __init__(self):
        self._plate_zone = -1
        self._coin_zone = -1
        self._backgroud_zone = -1
        self._brime_zone = -1
        self.findParts=getattr(self,"findParts")

    @property
    def plate_zone(self):
        if self._plate_zone == -1:
            self.findParts()
        return self._plate_zone

    @plate_zone.setter
    def plate_zone(self, value):
        self._plate_zone = value

    @property
    def coin_zone(self):
        if self._coin_zone == -1:
            self.findParts()
        return self._coin_zone

    @coin_zone.setter
    def coin_zone(self, value):
        self._coin_zone = value

    @property
    def backgroud_zone(self):
        if self._backgroud_zone == -1:
            self.findParts()
        return self._backgroud_zone

    @backgroud_zone.setter
    def backgroud_zone(self, value):
        self._backgroud_zone = value

    @property
    def brime_zone(self):
        if self._brime_zone == -1:
            self.findParts()
        return self._brime_zone

    @brime_zone.setter
    def brime_zone(self, value):
        self._brime_zone = value




class SampleImageProcessor(Parts):
    DEBUG = True
    DEBUG_PATH=None
    MIN_POINT_COUNT = 50  # 图像区域计算的最小面积
    _orgImage = np.array([])  #原始图像
    _diff_level = 5            #差异的阈值，默认为5
    _tagMap = None  # 保存联通结果的变量，这个就是整个的处理结果，因此非常有用
    _whiteColor=None #plate区域颜色的均值和方差
    _blackColor=None #plate区域颜色的均值和方差,None未设置
    _roiEllipse=None #plate区域的椭圆形
    _roiImage=None #roi区域的图像，会被处理，包括白平衡，黑平衡
    _roiAbsorbance=None #roi区域各点的吸光度值
    def __init__(self, image, diff_level=0):
        super().__init__()
        if diff_level<0:
            self.auto_diff_level=True
        else:
            self._diff_level = diff_level
            self.auto_diff_level = False
        self._orgImage = image


    def loadRaw(self,file_name):
        self._orgImage=np.load(file_name)

    def load_result(self,file_name):
        with open (file_name,"rb") as fp:
            self.tagMap=pickle.load(fp)
    @property
    def tagMap(self):
        if self._tagMap is None:
            self.connectedComponents()
        return self._tagMap

    @tagMap.setter
    def tagMap(self,value):
        self._tagMap=value


    def save_result(self,file_name):
        with open (file_name,"rb") as fp:
            pickle.dump(self.tagMap, fp)

    @property
    def bluredImage(self):
        # 这个参数已经经过优化好了，如果图像的对比对过小，可以降低sigmaColor的值，sigmaSpace的值基本可以了
        src=self._orgImage.astype(np.uint8)
        blur = cv.bilateralFilter(src=src, d=0, sigmaColor=30, sigmaSpace=9)
        return blur

    @property
    def diffImage(self, nFilterSize=5):
        """
        1、求x方向的微分
        2、求y方向的微分
        3、合并xy方向的微分值作为某点的微分值
            合并方法
        :param nFilterSize:
        :return: 微分以后的图像，(rows,cols,3)float32的数组
        每个通道的像素独自保存,
        最后一行和最后一列的数据没有意义
        实际上，对于5*5的数据，可以理解为应该有一个4*4的微分矩阵,因此在对其上，会普遍向左上角偏移一个点，但这不会造成问题
              1，2，3，4，5，
                1.2.3.4.
              1，2，3，4，5，
               .1.2.3.4.
              1，2，3，4，5，
               .1.2.3.4.
              1，2，3，4，5，
        """
        image = self.bluredImage
        # 进行积分
        cy = np.diff(image, 1, 0, append=0).astype(np.float32)
        cx = np.diff(image, 1, 1, append=0).astype(np.float32)
        # 进行一积分效果次增强
        if nFilterSize!=0:
            # kx =cv.getGaussianKernel(nFilterSize,1)
            # ky = cv.getGaussianKernel(nFilterSize,1).transpose()
            # print(kx)
            # print(ky)
            kx = np.ones((1, nFilterSize))
            ky = np.ones((nFilterSize, 1))
            cfy = cv.filter2D(cy, -1, ky)
            cfx = cv.filter2D(cx, -1, kx)
        else:
            cfy=cy
            cfx=cx

        # 这个是为了增强交叉点，但是效果不好
        # kx = np.array([1, 1, 1, 0, -1 - 1, -1])
        # ky = kx.transpose()
        # c2y = cv.filter2D(cy, -1, ky) / 7
        # c2x = cv.filter2D(cx, -1, kx) / 7

        def combine_xy(cx, cy):
            cxy = cy ** 2 + cx ** 2
            cxy = cxy[:-2, :-2, :] / 2
            cxy = np.sqrt(cxy)
            return cxy

        cxy = combine_xy(cfx, cfy)
        # cxy = cxy.astype(np.float32)
        # cxy = cxy.astype(np.uint8)
        # logging.info("积分图谱的最大值为：%f;最小值为：%f; 均值为：%s; 众数为：%f;"\
        #              %(
        #                 np.max(cxy),
        #                 np.min(cxy)),
        #                 str(np.mean(cxy,(0,1))),
        #                 1000)
        return cxy


    @property
    def diffGray(self):
        """
        支持了多通道的数据，也就是说，本操作支持超过3个色彩通道的数据，为将来的多通道采集做好准备
        :return: rows,cows float32的灰度图像，背景色为白色，就是培养皿和培养皿外，均为白色
        """
        diffImage = self.diffImage
        if len(diffImage.shape) >= 3:
            diff = np.sum(diffImage, axis=2)
        else:
            diff = diffImage
        return diff




    def diffBinary(self):
        """
        二值化的过程，由于使用微分差异值作为二值化的标准，很多问题得以避免
        二值化之后
        :return: rows,cols uint8数组，变异大的位置为0，变异小的位置为1
        """
        diffgray = self.diffGray
        rows, cols = diffgray.shape

        def autoDiffLevel(diffGray):
            counts, bins = np.histogram(diffGray, bins=255)
            max = bins[np.where(counts == np.max(counts[1:]))]
            return max*4
        if self.auto_diff_level:
            diff=autoDiffLevel(diffgray)
        else:
            diff = self._diff_level
        minLoc = np.where(diffgray > diff)
        diffBinary = np.ones((rows, cols), np.uint8) * 255
        diffBinary[minLoc] = 0
        return diffBinary

    def process(self):
        """
        处理图像，
        """

    def findSeed(self):
        """
        找到种子点的位置，也就是在培养皿的表面上随机的寻找一点，为floodfill进行参数准备
        :return:
        """

    def floodFill(self):
        """
        意义不大，暂时不实现，最大的好处是可以将小像素填充,降低mask的数据量
        使用这个更好的产生低噪音的diffgray
        :return:
        """
        # cv.floodFill(flood, mask, seedPoint=(1280, 900), newVal=0, loDiff=loDiff, upDiff=upDiff,
        #              flags=4 + cv.FLOODFILL_FIXED_RANGE)
        pass

    @property
    def image(self):
        return self._orgImage

    def connectedComponents(self):
        """
        寻找联通区域
        :return:
        """
        # 进行一次闭操作,一次开操作，降低产生的区域数量
        diffBin = self.diffBinary()
        kernel = np.ones((2, 2))
        # cv.dilate( )
        cv.morphologyEx(src=diffBin, dst=diffBin, op=cv.MORPH_CLOSE, kernel=kernel, iterations=2)
        # cv.morphologyEx(src=diffBin, dst=diffBin, op=cv.MORPH_OPEN, kernel=kernel, iterations=1)
        count, labels, status, centroids = cv.connectedComponentsWithStats(diffBin, connectivity=8)
        self.tagMap = ConnectedComponents(count, labels, status, centroids)
        # self.tag_map.
        # status x, y, h, w, s像素个数

        if self.DEBUG==False:
            STATUS_COLS_S = 4
            filtered = np.where(status[:, STATUS_COLS_S] > self.MIN_POINT_COUNT)
            outputdir =self.debugPath

            # 背景：
            def getStatus(status, i):
                x =status[i][0]
                y =status[i][1]
                w =status[i][2]
                h =status[i][3]
                s =status[i][4]
                return x, y, w, h, s

            for i in filtered[0]:
                # print(i, status[i], centroids[i])
                x, y, w, h, s = getStatus(status, i)
                # out_img = np.zeros((h, w), np.uint8)
                out_img = self._orgImage[y:y + h, x:x + w].copy()
                out_img_pos = np.where(labels[y:y + h, x:x + w] != i)
                out_img[out_img_pos] = 0
                filename = os.path.join(outputdir, "out" + str(i) + r".png")
                cv.imwrite(filename, out_img)
            filename = os.path.join(outputdir, "labels.png")
            cv.imwrite(filename, labels)
            filename = os.path.join(outputdir, "diffBin.png")
            cv.imwrite(filename, diffBin)
            filename = os.path.join(outputdir, "orgimage.png")
            cv.imwrite(filename, self._orgImage)
        return self.tagMap

    def findParts(self):
        """
        找到各个部位,这个有点麻烦，慢慢完善吧
        :return:
        """
        temp=list(self.tagMap.areaIndex[-10:])
        #删除掉第0张
        temp.remove(0)
        # print(self.tagMap.areaIndex[-10:])
        status = self.tagMap.status.data
        centroids = self.tagMap.centroids.data
        labels = self.tagMap.labels.data

        backgroud_zone=-1
        # plate_zone = -1
        plate_zone=-1
        coin_zone=-1


        def getStatus(status, i):
            x = status[i][0]
            y = status[i][1]
            w = status[i][2]
            h = status[i][3]
            s = status[i][4]
            return x, y, w, h, s

        def isBackgroudZone(i,index):
            #约束条件，慢慢来
            #面积最大的,并且是第二个找到的
            if 8-i==0 and index==1:
                return True
        def isPlateZone(i,index,width,height):
            # 约束条件，慢慢完善
            x, y, w, h, s = getStatus(status, index)
            x = x + w / 2
            y = y + h / 2
            # print("abs(x - width * 1 / 2)")
            # print(abs(x - width * 1 / 2))
            # print("abs(y - height * 1 / 2)")
            # print( abs(y - height * 1 / 2))
            # print( x, y, w, h, s)
            if abs(x - width * 1 / 2) < 200 \
                    and abs(y - height * 1 / 2) < 200 \
                    and w > width / 2 \
                    and h > height  / 2 \
                    and s > height * width / 4\
                    and x - w / 2 != 0:
                # print("found", x, y, w, h, s)
                return True
            else:
                return False

        def isCoinZone(i,index,width,height):
            # 约束条件，这个要马上实现
            x,y,w,h,s= getStatus(status, index)
            if x> width*2/3 and y>height*2/3 and w >height*1/25 and h>height*1/25 and s>height*width/100:
                # print("found",x,y,w,h,s)
                return True
        for i in range(0,len(temp)):
            index=temp[i]
            if isBackgroudZone(i,index):
                backgroud_zone=index
            elif isPlateZone(i,index,self.cols,self.rows):
                plate_zone=index
            elif isCoinZone(i,index,self.cols,self.rows):
                coin_zone=index

        #如果相应的区域没有找到，则返回-2
        if backgroud_zone==-1:
            backgroud_zone=-2
        if plate_zone == -1:
           plate_zone=-2
        if coin_zone==-1:
            coin_zone=-2

        self.backgroud_zone=backgroud_zone
        self.plate_zone=plate_zone
        self.coin_zone=coin_zone

        if self.DEBUG:
            STATUS_COLS_S = 4
            outputdir =self.debugPath
            # print(status)
            # 背景：

            def output(i,nametag):
                # print(i, status[i], centroids[i])
                x, y, w, h, s = getStatus(status, i)
                out_img = self._orgImage[y:y + h, x:x + w].copy()
                out_img_pos = np.where(labels[y:y + h, x:x + w] != i)
                out_img[out_img_pos] = 0
                filename = os.path.join(outputdir, nametag + str(i) + r".png")
                cv.imwrite(filename, out_img)

            for i in temp:
                output(i,str(i))

            output(self.backgroud_zone,"0backgroud_zone")
            output(self.coin_zone,"0coin_zone")
            output(self.plate_zone,"0plate_zone")
            # output(self.brime_zone,"brime_zone")

    def getMask(self, n):
        """
        通过联通区域序号，获得相应的mask
        :param n:
        :return:
        """
        mask=np.zeros((self.rows,self.cols),np.uint8)
        labels=self.tagMap.labels.data
        pos=np.where(labels==n)
        mask[pos]=255
        return mask

    @property
    def edge(self):
        """plate的"""
        plate_mask=self.getMask(self.plate_zone)
        plate_mask=255-plate_mask
        # cv.imshow("edge",plate_mask)
        # cv.waitKey(0)
        # countours,hierachy=cv.findContours(plate_mask,cv.RETR_EXTERNAL,method=cv.CHAIN_APPROX_NONE)
        return plate_mask


    @property
    def debugPath(self):
        if self.DEBUG_PATH is None:
            import time
            time_tag = time.ctime().replace(":", "_")
            # time_tag=""
            self.DEBUG_PATH=os.path.join(r"e:\desktop",time_tag)
            if not os.path.exists(self.DEBUG_PATH):
                os.makedirs(self.DEBUG_PATH)
        return self.DEBUG_PATH

    def findRoiEllipse(self):
        """
        自动查找ROI的椭圆
        :param img: 输入图像
        :return ellipse: 所拟合出的椭圆信息——((x,y),(a,b),angle)
                         其中x,y为椭圆中心点的位置，a,b为长短轴直径，angle为中心旋转角度
        todo:算法有问题，不能找到对合适的位置
        """
        # 转换编码格式
        img = self.getMask(self.plate_zone)
        # 结构元素，用于腐蚀膨胀
        element = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
        # 对二值图像，腐蚀后膨胀
        erode = cv.erode(img, element)
        dilate = cv.dilate(erode, element)
        # 查找轮廓
        contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # 拟合ROI内圈
        for contour in contours:
            # 滤掉周长小的轮廓
            if len(contour) > self.rows:
                # 对轮廓凸包计算
                contour = cv.convexHull(contour)
                # 拟合ROI轮廓信息，严格意义上为椭圆
                ellipse = cv.fitEllipse(contour)
                ellipse=Ellipse(ellipse)
                # ellipse.enlarge(1)
                if self.DEBUG:
                    filename=os.path.join(self.debugPath,"ellipse.png")
                    img=cv.circle(self.image,ellipse.centerInt,ellipse.radiusInt,color=(255,0,0),thickness=1)
                    cv.imwrite(filename,img)
                return ellipse


    @property
    def roiEllipse(self):
        if self._roiEllipse is None:
            self._roiEllipse=self.findRoiEllipse()
        return self._roiEllipse

    @property
    def roiBackgroudImage(self):
        """
        A=-lg(I/I0) I0=T100
        这个的目的是为了计算菌落的T100，用来进行处理数据，获得每个点的透光率使用
        :return:
        """
        logging.debug("start mask")
        plate_mask = self.getMask(self.plate_zone)   #50ms
        logging.debug("end mask")
        src=self._orgImage.astype(np.uint8)
        roiBackgroudImage=cv.bitwise_and(src, cv.cvtColor(plate_mask, cv.COLOR_GRAY2BGR)) #10ms
        logging.debug("end bit and")
        roiBackgroudImage=cv.morphologyEx(roiBackgroudImage,
                                          op=cv.MORPH_DILATE,
                                          kernel=np.ones((5,5),np.uint8),
                                          iterations=30
                                          )  #50ms
        logging.info("end morph")
        if self.DEBUG:
            filename=os.path.join(self.debugPath,"roiBackgroudImage.jpg")
            cv.imwrite(filename, roiBackgroudImage)
        return roiBackgroudImage

    @property
    def rows(self):
        return self._orgImage.shape[0]

    @property
    def cols(self):
        return self._orgImage.shape[1]

    @property
    def height(self):
        return self._orgImage.shape[0]

    @property
    def width(self):
        return self._orgImage.shape[1]

    @property
    def whiteColor(self):
        if self._whiteColor is not None:
            return self._whiteColor
        if self.plate_zone==-1:
            self.findParts()
        mask=self.getMask(self.plate_zone)
        mean,stdev=cv.meanStdDev(self.image,mask=mask)
        self._whiteColor=ColorSpace(mean,stdev)
        logging.info("本样品的白区域为：mean：;std%s：%s。"%(str(mean),str(stdev)))
        return self._whiteColor

    @property
    def blackColor(self):
        if self._blackColor is not None:
            return self._blackColor

        if self.coin_zone==-1:
            self.findParts()

        if self.coin_zone>0:
            mask=self.getMask(self.coin_zone)
            mean,stdev=cv.meanStdDev(self.image,mask=mask)
            self._blackColor=ColorSpace(mean,stdev)
        else:
            #其他设置黑色区域的方法
            mean=np.zeros((3,1))
            stdev=np.zeros((3,1))
            self._blackColor=ColorSpace(mean,stdev)
        logging.info("本样品的黑色区域为：mean：;std%s：%s。"%(str(self._blackColor.mean),str(self._blackColor.std)))
        return self._blackColor



    def readRaw(self, filename):
        self._orgImage = np.load(filename)
        
    @property
    def roiMask(self):
        mask=np.zeros((self.rows,self.cols),np.uint8)
        ellipse=self.roiEllipse
        mask = cv.circle(mask, ellipse.centerInt, ellipse.radiusInt-5, color=255, thickness=-1)
        return mask
            
    @property
    def roiImage(self):
        if self._roiImage is None:
            pos=np.where(self.roiMask!=255)
            self._roiImage=self._orgImage.copy()
            self._roiImage[pos]=0
        return self._roiImage

    def blackBalance(self):
        black = np.transpose(self.blackColor.mean)
        cvmat=self.roiImage
        cvmat = cvmat -black
        cvmat = np.clip(cvmat, 0, None)
        self._roiImage=cvmat
        return self._roiImage
    
    def whiteBalance(self):
        debug=True
        """
        对数校正以后进行白平衡
        :return:
        """
        white_color=self.whiteColor.mean
        B,G,R=cv.split(self.roiImage)
        b,g,r=white_color
        B = B/b*255
        G = G/g*255
        R = R/r*255
        self._roiImage=cv.merge(np.clip((B,G,R),0,255))

        if self.DEBUG and debug:
            self._roiImage = np.array(self._roiImage, np.uint8)
            filename=os.path.join(self.debugPath,"roiImage.png")
            cv.imwrite(filename,self.roiImage)
        # self.cvmat=np.array(self.cvmat,np.uint8)
    def calAbsorbanceImage(self):
        I0=self.whiteColor.mean.transpose()-self.blackColor.mean.transpose()
        I=self._orgImage-self.blackColor.mean.transpose()
        # I0=np.clip(I0,1,255)
        I=np.clip(I,1,255)
        absorbance=np.log10(I0)-np.log10(I)
        absorbance=absorbance*lightness
        absorbance = np.clip(absorbance, 0, 255)
        return absorbance

        if self.DEBUG:
            absorbance = np.array(absorbance, np.uint8)
            # cv.imshow("test", absorbance)
            # cv.waitKey(0)
            filename=os.path.join(self.debugPath,"calAbsorbanceImage.png")
            cv.imwrite(filename,absorbance)


    def calAbsorbanceImage_(self):
        I0 = self.whiteColor.mean.transpose() - self.blackColor.mean.transpose()
        I = self._orgImage - self.blackColor.mean.transpose()

        absorbance = np.log10(I0) - np.log10(I)
        absorbance = absorbance * lightness

        return absorbance

    def scale(self,image):
        min=np.min(image)
        max=np.max(image)
        scaled=(image-min)/(max-min)*255
        return scaled

    def findContours(self):
        image=self.edge
        image=(255-image).astype(np.uint8)
        contours,hierachy=cv.findContours(image,mode=cv.RETR_TREE,method=cv.CHAIN_APPROX_NONE)

        if self.DEBUG:
            # blank=np.zeros(self._roiImage.shape,np.uint8)
            # image=cv.drawContours(outImage,contours,-1,255,1)
            outImage = cv.drawContours(self.image.copy(), contours, -1, (255, 0, 0), 1, maxLevel=2)
            filename=os.path.join(self.debugPath,"contours.png")
            cv.imwrite(filename,outImage)
            return outImage
        return contours

    def processRGB48(self,fileName=None):
        fileName=r"D:\temp\Capture\S.aureus__rgb.png"
        image_uint16=cv.imread(fileName,cv.IMREAD_UNCHANGED)
        image_uint16=image_uint16/256
        processor = SampleImageProcessor(image_uint16, 5)
        image = processor.calAbsorbanceImage()
        image = image.astype(np.uint8)
        basename = os.path.basename(fileName)
        segs=basename.split(".")
        eng_tag=len(segs[-1])
        basename=str(basename[:-eng_tag-1])
        outPath=os.path.dirname(fileName)
        out = os.path.join(outPath, basename + "0.png")
        i = 0
        while os.path.exists(out):
            i = i + 1
            out = os.path.join(outPath, basename + str(i) + ".png")
        cv.imwrite(out, image)


    def processNpy(self,filePath=r"\\dsd.yb93.cn\softwares\ColonyCounting\TestData\adaption\Images"):
        """
        第一步：读取数据
        第二步：计算黑平衡
        第三步：计算黑平衡差值
        第四步：加上1以后,计算对数值
        第五步：将对数值map成1~255
        第六步：保存成图片，名称为原来图片+raw.png
        """
        # # self.load_file()
        # # self.readRaw(filename)
        # self.blackBalance()
        # self.whiteBalance()
        # # self.findContours()
        # # self.calAbsorbanceImage()
        # #
        # #
        # #
        # # self.cal_black_balance()
        # # self.calibrate_black_balance()
        # # self.log_transform()
        # # self.calibrate_white_balance()
        # # self.map_image()
        # # self.save_image()

        filePath = des_path
        filePath=filePath+r"\*.npy"
        import glob
        files=glob.glob(filePath)
        outPath=out_path
        for file in files:
            print(file)
            image=np.load(file)/100
            processor=SampleImageProcessor(image,5)
            # processor.connectedComponents()
            # processor.findParts()
            # print(processor.whiteColor)
            # print(processor.blackColor)
            # bg=processor.roiBackgroudImage
            # print(processor.findRoiEllipse())
            image=processor.calAbsorbanceImage()
            image =image.astype(np.uint8)
            basename=os.path.basename(file)
            out=os.path.join(outPath,basename+"0.png")
            i=0
            while os.path.exists(out):
                i=i+1
                out=os.path.join(outPath,basename+str(i)+".png")

            cv.imwrite(out,image)

    def processDir(self,filePath=r"\\dsd.yb93.cn\softwares\ColonyCounting\TestData\adaption\Images"):
        # filePath=r"\\dsd.yb93.cn\softwares\ColonyCounting\AccByInstrument\Images\**"
        filePath = des_path
        # filePath=r"D:\temp\output"
        filePath = filePath+r"\*.png"
        import glob
        files=glob.glob(filePath,recursive=True)
        outPath=r"e:\desktop"
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        for file in files:
            # print(file)
            image=cv.imread(file)
            processor=SampleImageProcessor(image,5)
            image=processor.findContours()
            image =image.astype(np.uint8)
            basename=os.path.basename(file)
            out=os.path.join(outPath,basename+"0.png")
            i=0
            while os.path.exists(out):
                i=i+1
                out=os.path.join(outPath,basename+str(i)+".png")
            cv.imwrite(out,image)



if __name__ == "__main__":
    # filename=r"Y:\ColonyCounting\TestData\test-0419\Images\E.coli-0330-100-2-coin.png"
    # filename = r"C:\work\ColonyMarking\testCodes\input\002.jpg"
    # # filename = r"Y:\ColonyCounting\TestData\test-0419\Images\E.coli-0330-100-2.png"
    filename = r"e:\desktop\E.coli\E.coli06.png"
    # filename = r"C:/Users/qiaos/Documents/Tencent Files/2949362/FileRecv/100Thu Mar 10 17-43-55 2022.png"
    # filename = r"Y:/ColonyCounting/TestData/test-0419\Images\L.monocytogenes-0408-150-2.png"
    # filename = r"Y:/ColonyCounting/TestData/test-0419\Images\Salmonella-0330-100-1-coin.png"
    # filename =r"Y:/ColonyCounting/TestData/test-0419\Images\Salmonella-0330-100-2.png"
    # filename = r"Y:/ColonyCounting/TestData/test-0419\Images\Shigella-0330-100-3-coin.png"



    base = cv.imread(filename)
    # base=filter_image(base)
    sample_image = SampleImageProcessor(image=base, diff_level=5)
    # sample_image.processDir(r"\\dsd.yb93.cn\softwares\ColonyCounting\TestData\test-0419\Images")
    # aa=sample_image.connectedComponents()
    # sample_image.findParts()
    # flood_diff(base)
    # sample_image.processRGB48()
    # sample_image.processDir()
    sample_image.processNpy(r'E:\Desktop\E.coli')