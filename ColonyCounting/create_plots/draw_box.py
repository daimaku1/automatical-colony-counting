# 开发者：热心网友
# 开发时间：2022/11/18 12:18
# coding:utf-8
import glob
import os.path
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from JunLuoCounting.model_src.u2net.src import u2net_full
from torchvision.models import resnet50

from JunLuoCounting.model_src.u2net.pred_api import main as u2net_pred
from JunLuoCounting.model_src.resnet__50.pred_api import main as resnet_pred

from create_mask_after_disk_markiing import create_disk_single

edge_thresh = 0.1
threshold = 0.9
regular_size = 128
support_format = ['jpg', 'png']
weight_path = r'F:\final_version\final_version\JunLuoCounting\weights'
infer_dir = r'E:\Desktop\testout'
exps_dir = r'F:\final_version\final_version\JunLuoCounting\Inferences'
device = torch.device('cpu')


def prepare_mdoels(u2net_cir_weights,
                   u2net_colony_weights,
                   resnet_weights):
    model_1 = u2net_full().to(device)
    assert os.path.exists(u2net_cir_weights), "file: '{}' dose not exist.".format(u2net_cir_weights)
    weights = torch.load(u2net_cir_weights, map_location=device)
    if "model" in weights:
        model_1.load_state_dict(weights["model"])
    else:
        model_1.load_state_dict(weights)

    model_2 = u2net_full().to(device)
    assert os.path.exists(u2net_colony_weights), "file: '{}' dose not exist.".format(u2net_colony_weights)
    weights = torch.load(u2net_colony_weights, map_location=device)
    if "model" in weights:
        model_2.load_state_dict(weights["model"])
    else:
        model_2.load_state_dict(weights)

    model_3 = resnet50(num_classes=10).to(device)
    # load model weights
    assert os.path.exists(resnet_weights), "file: '{}' dose not exist.".format(resnet_weights)
    model_3.load_state_dict(torch.load(resnet_weights, map_location=device))

    return model_1, model_2.to(device), model_3.to(device)


def get_input(img_dir):
    imgs = []
    for img_form in support_format:
        imgs.extend(glob.glob(img_dir + '/*.' + img_form))
    return imgs


def remove_edge(img_paths, model, remove=True):
    if not remove:
        imgs = []
        for img_path in img_paths:
            img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            imgs.append(img)
        return imgs

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1280,961)),
    ])

    imgs = []
    for img_path in img_paths:
        origin, mask = u2net_pred(model=model, origin_img=img_path, transform=data_transform)
        # cv2.imshow('circle', cv2.resize(mask, dsize=None, fx=0.2, fy=0.2))
        # cv2.imshow('origin', cv2.resize(origin, dsize=None, fx=0.2, fy=0.2))
        mask_thre = (np.where(mask > edge_thresh, 1, 0) * 255).astype('uint8')
        # cv2.imshow('mask', cv2.resize(mask_thre, dsize=None, fx=0.5, fy=0.5))
        # mask_thre = (1 - (mask_thre / 255)).astype('uint8')
        # mask_thre = np.expand_dims(mask_thre, -1)
        # mask_thre = np.repeat(mask_thre.copy(), origin.shape[-1], axis=-1)
        # origin = np.array(origin, dtype=np.uint8)
        # img = origin * mask_thre
        img = create_disk_single(mask_thre, origin, edge_thresh)
        imgs.append(img)
        # cv2.imshow('img', cv2.resize(origin, dsize=None, fx=0.5, fy=0.5))
        # cv2.imshow('res', cv2.resize(img, dsize=None, fx=0.5, fy=0.5))
        # cv2.waitKey()
        # cv2.imshow('img_without_circle', cv2.resize(img, dsize=None, fx=0.2, fy=0.2))
        # cv2.waitKey()
    return imgs


def u2net_predict(imgs, model):
    oris, masks = [], []

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1280,961)),
    ])

    for img in imgs:
        origin, mask = u2net_pred(model=model, origin_img=img, transform=data_transform)
        mask_thre = (np.where(mask > threshold, 1, 0) * 255).astype('uint8')
        origin = np.array(origin, dtype=np.uint8)
        oris.append(origin)
        masks.append(mask_thre)
        # cv2.imshow('mask', cv2.resize(mask, dsize=None, fx=0.2, fy=0.2))
        # cv2.imshow('mask_thre', cv2.resize(mask_thre, dsize=None, fx=0.2, fy=0.2))
        # cv2.waitKey()

    return oris, masks


def create_target_image(img, mask, model, revolve=True):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    labels = np.expand_dims(labels, 2).repeat(3, axis=2)
    im = img.copy()
    cnt = 0
    for label, (x, y, w, h, s) in enumerate(stats):
        if label == 0:
            continue
        # (x, y), (x + w, y + h)
        if 1 in (x, y, w, h, s):
            continue
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
        segment = (labels == label) * img
        segment = segment[y: y + h, x: x + w]
        if revolve:
            segment = revolve_image(segment)
        segment = resize2regular(segment, regular_size)
        preds = resnet_predict([segment], model)
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f'co: {preds[0]}'
        tx1, ty1 = x, y - 20
        tx2, ty2 = x + 128, y
        org = (tx1 + 5, ty2 - 5)
        fontScale = 0.8
        color = (255, 255, 255)  # 白色
        thickness = 1
        cv2.putText(im, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

        cnt += preds[0]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return im, cnt


def save_res(save_dir, imgs, names):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for img, name in zip(imgs, names):
        cv2.imwrite(os.path.join(save_dir, name), img)


def resnet_predict(imgs, model):

    data_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.1, 0.1, 0.1),
            #                      std=(0.1, 0.1, 0.1))
         ])

    preds = []
    for img in imgs:
        pred = int(resnet_pred(model, img, transform=data_transform))
        preds.append(pred)
    return preds


def resize2regular(img, dsize: int):
    assert len(img.shape) >= 2, 'abnormal seg-image shape'
    assert 1 not in img.shape
    h, w = img.shape[:2]
    if max([w, h]) <= 128:
        padding_top = int((dsize - h) / 2)
        padding_left = int((dsize - w) / 2)
        img = cv2.copyMakeBorder(img,
                                 padding_top, dsize - h - padding_top,
                                 padding_left, dsize - w - padding_left,
                                 cv2.BORDER_CONSTANT, value=(0,))
    else:
        ratio = dsize / max([w, h])
        img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
        h, w = img.shape[:2]
        padding_top = int((dsize - h) / 2)
        padding_left = int((dsize - w) / 2)
        img = cv2.copyMakeBorder(img,
                                 padding_top, dsize - h - padding_top,
                                 padding_left, dsize - w - padding_left,
                                 cv2.BORDER_CONSTANT, value=(0, ))
    return img


def revolve_image(img):
    def find_cnt_ellipse(image, factor=True):
        """
        自动查找ROI的椭圆与轮廓
        :param image: 输入图像
        :param factor: 是否同时得到椭圆和轮廓
        :return: roi_ellipse: 所拟合出的椭圆信息——((x,y),(a,b),angle)
                              其中x,y为椭圆中心点的位置，a,b为长短轴直径，angle为中心以a顺时针旋转角度
                 roi_contour: ROI轮廓点集
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype('uint8')
        # 查找轮廓
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 拟合内圈
        roi_ellipse = ()
        roi_contour = ()
        max_points = 0
        for cnt in contours:
            # 保留周长最大的轮廓
            if len(cnt) > max_points:
                max_points = len(cnt)
                # 对轮廓凸包计算
                cnt = cv2.convexHull(cnt)
                try:
                    if factor:
                        # 拟合轮廓信息，严格意义上为椭圆
                        roi_ellipse = cv2.fitEllipse(cnt)
                        roi_contour = cnt
                    else:
                        roi_contour = cnt
                except:
                    continue
        if factor:
            return roi_ellipse, roi_contour
        else:
            return roi_contour
    ellipse, contour = find_cnt_ellipse(img, factor=True)
    if len(ellipse) == 0 or len(contour) == 0:
        return img
    else:
        rows, cols, channels = img.shape
        a = ellipse[1][0]
        b = ellipse[1][1]
        angle = ellipse[2]
        padding = max(rows, cols)
        tmp = cv2.copyMakeBorder(img,
                                 int(padding / 4), int(padding / 4),
                                 int(padding / 4), int(padding / 4),
                                 cv2.BORDER_CONSTANT, value=(0,))
        rows, cols, channels = tmp.shape
        # 旋转变换
        if a > b:
            # getRotationMatrix2D有三个参数，第一个为旋转中心，第二个为旋转角度(逆时针)，第三个为缩放比例
            M1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle - 90, 1)  # a为长轴
        else:
            M1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # a为短轴
        res = cv2.warpAffine(tmp, M1, (cols, rows))
        try:
            new_contour = find_cnt_ellipse(res, factor=False)
            # 获取轮廓外接框
            x, y, w, h = cv2.boundingRect(new_contour)
            # x_ = cols / 2 - (x + w / 2)
            # y_ = rows / 2 - (y + h / 2)
            # # 平移变换
            # M2 = np.float32([[1, 0, x_], [0, 1, y_]])
            # # cv.warpAffine()第三个参数为输出的图像大小，值得注意的是该参数形式为(width, height)
            # dst = cv2.warpAffine(res, M2, (cols, rows))
            dst = res[y: y + h + 5, x: x + w + 5, ]
            # 求质心
            # end_contour = find_cnt_ellipse(dst, factor=False)
            # M = cv2.moments(end_contour)
            # cx = int(M['m10'] / M['m00'])
            # cy = int(M['m01'] / M['m00'])
            # if cy < rows / 2:
            #     dst = cv2.flip(dst, 0)
            # if cx < cols / 2:
            #     dst = cv2.flip(dst, 1)
            return dst
        except:
            return img


def main():
    u2net_weights_dir = os.path.join(weight_path, 'u2net')
    resnet_weight_dir = os.path.join(weight_path, 'ResNet')
    # u1 = 'model_5_circle_without_resize_1_10.pth'
    u1 = 'model_Circle_preformance.pth'
    u2 = 'model_best_direct_1_10.pth'
    r = 'resnet_50_new.pth'

    u1 = os.path.join(u2net_weights_dir, u1)
    u2 = os.path.join(u2net_weights_dir, u2)
    r = os.path.join(resnet_weight_dir, r)

    u1, u2, r = prepare_mdoels(u1, u2, r)

    img_dir = os.path.join(infer_dir, 'image')
    exp_dir = os.path.join(exps_dir, 'predict_res')

    cdir, dirs, _ = next(os.walk(exp_dir))
    indices = [int(dir.replace('exp_', '')) for dir in dirs]
    if not indices:
        indices.append(-1)
    save_dir = os.path.join(cdir, f'exp_{max(indices) + 1}')

    img_paths = get_input(img_dir)

    print('Start inference ')
    for img_path in img_paths:
        time_start = time.time()
        print(f'Inferencing {os.path.split(img_path)[-1]}')
        imgs = remove_edge([img_path], u1, remove=True)
        imgs, masks = u2net_predict(imgs, u2)
        target_img, colonies = create_target_image(imgs[0], masks[0], r, revolve=True)
        img_name = f'{colonies}_' + os.path.split(img_path)[-1]
        # # save_res(save_dir, seg_imgs, [f'{preds[i]}_{i}_' + img_name for i in range(len(seg_imgs))])
        save_res(save_dir, [target_img], [img_name])
        print(fr'total time: {time.time() - time_start}')

if __name__ == '__main__':
    main()
