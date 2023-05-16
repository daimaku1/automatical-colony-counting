import glob
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms

from src import u2net_full, u2net_lite


def time_synchronized():
    # torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def predict_output(img_path = r"./test.png",weights_path = r"E:\Desktop\model_best.pth"):
    sample_name=os.path.basename(img_path)[0:-4]
    outdir=os.path.join(os.path.dirname(img_path),"output")
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            pass
    maskfile=os.path.join(outdir,sample_name+"_mask.png")
    outputfile=os.path.join(outdir,sample_name+"_output.png")
    threshold = 0.9
    assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1280,961)),
        # transforms.Normalize(mean=(1, 1, 1),
        #                      std=(0.1, 0.1, 0.1))
    ])

    origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # origin_img=origin_img.astype(np)

    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    model = u2net_full()
    weights = torch.load(weights_path, map_location=device)
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        # model(init_img)
        t_start = time_synchronized()
        pred = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        # pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.where(pred > threshold)
        # by cao
        # cv2.imshow('pred', (cv2.resize(pred, dsize=None, fx=1, fy=1) * 255).astype(np.uint8))
        # cv2.waitKey(0)
        # cv2.imwrite('./pred.png', (pred * 255).astype(np.uint8))
        #
        r,g,b=cv2.split(origin_img)
        r[pred_mask] = 0
        g[pred_mask] = 255
        b[pred_mask] = 0
        outimage=cv2.merge((b,g,r))

        mask=np.zeros((h, w ),np.uint8)
        mask[pred_mask]=255

        # origin_img[:,:,0]=pred_mask
        # seg_img = origin_img * pred_mask[..., None]
        cv2.imwrite(maskfile, mask)
        cv2.imwrite(outputfile,outimage)

        # cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

def predict(img_path = r"./test.png",weights_path = r"save_weights\model_89.pth"):
    sample_name=os.path.basename(img_path)[:-4]
    outdir=os.path.join(os.path.dirname(img_path),"output")
    maskfile=os.path.join(outdir,sample_name+"_mask.png")
    outputfile=os.path.join(outdir,sample_name+"_output.png")
    threshold = 0.01
    assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(1024),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                      std=(0.229, 0.224, 0.225))
    ])

    origin_img = cv2.cvtColor(cv2.imread(img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # origin_img=origin_img.astype(np)

    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

    model = u2net_full()
    weights = torch.load(weights_path, map_location=device)
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = model(img)
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))
        pred = torch.squeeze(pred).to("cpu").numpy()  # [1, 1, H, W] -> [H, W]
        pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        pred_mask = np.where(pred > threshold, 1, 0)
        # by cao
        cv2.imshow('pred', (cv2.resize(pred_mask, dsize=None, fx=1, fy=1) * 255).astype(np.uint8))
        cv2.waitKey(0)
        cv2.imwrite('./pred.png', (pred * 255).astype(np.uint8))
        #
        origin_img = np.array(origin_img, dtype=np.uint8)
        seg_img = origin_img * pred_mask[..., None]
        plt.imshow(seg_img)
        plt.show()
        cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():

    # weights_path =r"C:\work\deep-learning-for-image-processing\pytorch_segmentation\u2net\u2net_full.pth"
    # img_path = "./test.jpg"  # default = test.png
    pred_dir=r"E:\Desktop\testout\image"
    files=glob.glob(pred_dir + '/*.png')
    for file in files:
        predict_output(file)


if __name__ == '__main__':
    main()
