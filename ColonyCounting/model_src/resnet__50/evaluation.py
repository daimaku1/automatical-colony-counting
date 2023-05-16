import glob
import os
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from final_version.JunLuoCounting.model_src.u2net.src.model import u2net_full
from sklearn.metrics import f1_score
from torchvision.models import resnet50
from final_version.JunLuoCounting.model_src.u2net.pred_api import main as u2net_pred
from sklearn.metrics import confusion_matrix


img_dir = r"E:\Code_versions\final_version\JunLuoCounting\model_src\resnet__50\data\train"
weights_path = r"E:\Code_versions\final_version\JunLuoCounting\weights"
json_path = './class_indices.json'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


def remove_edge(imgs, remove=True) -> []:
    if not remove:
        return imgs
    model = u2net_full()
    weights = torch.load(os.path.join(weights_path, 'u2net', 'model_5_circle_without_resize_1_10.pth'), map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024),
    ])

    for img in imgs:
        origin, mask = u2net_pred(model=model, origin_img=img, transform=data_transform)
        # cv2.imshow('circle', cv2.resize(mask, dsize=None, fx=0.2, fy=0.2))
        # cv2.imshow('origin', cv2.resize(origin, dsize=None, fx=0.2, fy=0.2))
        mask_thre = (np.where(mask > 0.2, 1, 0) * 255).astype('uint8')
        mask_thre = (1 - (mask_thre / 255)).astype('uint8')
        mask_thre = np.expand_dims(mask_thre, -1)
        mask_thre = np.repeat(mask_thre.copy(), origin.shape[-1], axis=-1)
        origin = np.array(origin, dtype=np.uint8)
        img = origin * mask_thre
        imgs.append(img)
        # cv2.imshow('img_without_circle', cv2.resize(img, dsize=None, fx=0.2, fy=0.2))
        # cv2.waitKey()
    return imgs


def u2net_predict(imgs):
    model = u2net_full()
    weights = torch.load(os.path.join(weights_path, 'u2net', 'model_5_circle_without_resize_1_10.pth'), map_location='cpu')
    if "model" in weights:
        model.load_state_dict(weights["model"])
    else:
        model.load_state_dict(weights)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024),
    ])

    for img in imgs:
        origin, mask = u2net_pred(model=model, origin_img=img, transform=data_transform)
        # cv2.imshow('circle', cv2.resize(mask, dsize=None, fx=0.2, fy=0.2))
        # cv2.imshow('origin', cv2.resize(origin, dsize=None, fx=0.2, fy=0.2))
        mask_thre = (np.where(mask > 0.2, 1, 0) * 255).astype('uint8')
        mask_thre = (1 - (mask_thre / 255)).astype('uint8')
        mask_thre = np.expand_dims(mask_thre, -1)
        mask_thre = np.repeat(mask_thre.copy(), origin.shape[-1], axis=-1)
        origin = np.array(origin, dtype=np.uint8)
        img = origin * mask_thre
        imgs.append(img)
        # cv2.imshow('img_without_circle', cv2.resize(img, dsize=None, fx=0.2, fy=0.2))
        # cv2.waitKey()
    return imgs


def main():
    # create model
    model = resnet50(num_classes=10).to(device)

    # load model weights
    weights = os.path.join(weights_path, 'ResNet', 'resnet_50_50epo.pth')
    assert os.path.exists(weights), "file: '{}' dose not exist.".format(weights)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    # read class_indict
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    preds = []
    labels = []
    for cdir, _, files in os.walk(img_dir):
        # cnt = 0

        for file in files[:200 if len(files) >= 200 else len(files)]:
            img_path = os.path.join(cdir, file)
            label = int(os.path.splitext(file)[0][-1])
            labels.append(label)

            data_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])

            # load image
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            # plt.imshow(img)
            # [N, C, H, W]
            img_ = data_transform(img)
            # expand batch dimension
            img_ = torch.unsqueeze(img_, dim=0)
            # prediction
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img_.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
                preds.append(int(class_indict[str(predict_cla)]))
        # if len(preds):
        #     labels = np.array(labels)
        #     preds = np.array(preds)
        #
        #     acc = np.equal(preds, labels).sum().item() / len(labels)
        #     micro_f1 = f1_score(labels, preds, average='micro')
        #     macro_f1 = f1_score(labels, preds, average='macro')
        #     print('[model % s] predict %s val_accuracy: %.3f  val_micro_f1: %.3f val_macro_f1: %.3f' %
        #           (weights_path, labels[0], acc, micro_f1, macro_f1))
            # img = np.array(img)
            # cv2.imwrite(f'./temp/{predict_cla}_{cnt}.png', img)
            # cnt += 1
    matrix = confusion_matrix(labels, preds)
    for i in matrix:
        print(i)

if __name__ == '__main__':
    main()