import os
import cv2
import numpy as np
import glob

input_dir=r"C:\work\final_version\JunLuoCounting\Inferences\predict_res\exp_24"
#输入是finddisk的u2net的输出，0~255

# org_dir=r"C:\work\JunLuoCount-master\Inferences\image"


def create_disk_single(edge_mask, ori_image, thresh):
    mask = np.zeros(edge_mask.shape[0:2], np.uint8)
    fore_pos = np.where(edge_mask[:, :] > thresh)
    # #有大约3个像素的左上角的偏移，这个是修正
    # fore_pos0 = fore_pos[0] + shift_y
    # fore_pos1 = fore_pos[1] + shift_x
    # fore_pos0 = np.clip(fore_pos0, 0, edge_mask.shape[0])
    # fore_pos1 = np.clip(fore_pos1, 0, edge_mask.shape[1])
    # fore_pos = (fore_pos0, fore_pos1)
    # 生成2值图像
    mask[fore_pos] = 255
    outer = None
    inner = None
    # 查找边缘
    cnts, hyrs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        center, radius = cv2.minEnclosingCircle(cnt)

        if abs(center[0] - 1280) < 100 and abs(center[1] - 961) < 100:  # 圆心不是太偏
            # 长得像个圆形
            area = cv2.contourArea(cnt)
            circle_area = np.pi * radius ** 2
            ratio = abs(np.log(area / circle_area))
            if ratio < np.log(1.1):  # 长得像个圆形
                # 逻辑不严谨，但是能用
                if inner is None:  # 先设定外圈
                    inner = cnt
                elif outer is None:  # 后设定内圈
                    outer = cnt
    if inner is not None:
        final_mask = np.zeros(edge_mask.shape[0:2], np.uint8)
        cv2.drawContours(final_mask, [inner], -1, 1, -1)
        final_roi = np.multiply(cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR), ori_image)
        return final_roi
    else:
        # cv2.imshow('error image', ori_image)
        # cv2.imshow('error mask', mask)
        # cv2.waitKey(0)
        return None


if __name__ == '__main__':


    files=glob.glob(input_dir+r"\*.png")
    for n in range(1, 2):
        thresh=0.1*n
        shift_x=0
        shift_y=0
        output_dir = input_dir+r"\output"+str(thresh)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file in files:
            basename=os.path.basename(file)
            # org_file=os.path.join(org_dir,basename)
            output_name=os.path.join(output_dir,basename)
            resimg=cv2.imread(file)
            mask=np.zeros(resimg.shape[0:2],np.uint8)
            fore_pos=np.where(resimg[:,:,0] > thresh * 255 )
            # #有大约3个像素的左上角的偏移，这个是修正
            # fore_pos0 = fore_pos[0] + shift_y
            # fore_pos1 = fore_pos[1] + shift_x
            # fore_pos0 = np.clip(fore_pos0, 0, resimg.shape[0])
            # fore_pos1 = np.clip(fore_pos1, 0, resimg.shape[1])
            # fore_pos = (fore_pos0, fore_pos1)
            #生成2值图像
            mask[fore_pos]=255
            outer=None
            inner=None
            #查找边缘
            cnts,hyrs=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                center,radius=cv2.minEnclosingCircle(cnt)

                if abs(center[0]-1280)<100 and abs(center[1]-961)<100:# 圆心不是太偏
                    #长得像个圆形
                    area=cv2.contourArea(cnt)
                    circle_area=np.pi*radius**2
                    ratio=abs(np.log(area/circle_area))
                    if ratio<np.log(1.1): #长得像个圆形
                        #逻辑不严谨，但是能用
                        if inner is None: #先设定外圈
                            inner=cnt
                        elif outer is None: #后设定内圈
                            outer=cnt
            if inner is not None:
                final_mask=np.zeros(resimg.shape[0:2],np.uint8)
                cv2.drawContours(final_mask,[inner],-1,255,-1)
                cv2.imwrite(output_name,final_mask)


