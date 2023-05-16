import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

img_dir = r'E:\Datasets\Circle\DUTS-TR\DUTS-TR-Mask'

img_names = glob.glob(img_dir + '/*.png')

area_colony = 0
area_img = 0
ratio_list = []

for img_name in img_names:
    img = cv2.imread(img_name, 0)
    img = np.where(img != 0, 1, 0).astype('uint8')
    area_colony = np.sum(img)
    area_img = img.shape[0] * img.shape[1]
    ratio_list.append(round(area_colony / area_img, 3))

ratio_list = np.array(ratio_list)

#准备绘制数据
x = np.unique(ratio_list)
y = [np.sum(np.where(ratio_list == i, 1, 0)) for i in np.unique(ratio_list)]

width = 0.0003
# "g" 表示红色，marksize用来设置'D'菱形的大小
plt.bar(x, y, width, label="num of pic", color='b')
#绘制坐标轴标签
plt.xlabel("Proportion of area of edge")
plt.ylabel("num of pic")
# plt.plot(x, y, marker = "v",markersize=4, markerfacecolor='r', color='orange')
# plt.xticks(np.arange(0, 0.1, 0.01))
#显示图例
plt.legend(loc="lower right")
#调用 text()在图像上绘制注释文本
#x1、y1表示文本所处坐标位置，ha参数控制水平对齐方式, va控制垂直对齐方式，str(y1)表示要绘制的文本
# for x1, y1 in zip(x, y):
#     plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
#保存图片
plt.savefig("2.jpg")
plt.show()
