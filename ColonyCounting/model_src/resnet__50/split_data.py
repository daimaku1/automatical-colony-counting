# coding:utf-8
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
import os, cv2


def get_img_label(data_path):
    cdir, _, files = next(os.walk(data_path))
    img_paths = []
    labels = []
    for file in files:
        if file.endswith('.png'):
            label = os.path.splitext(file)[0][-1]
            img_path = os.path.join(cdir, file)
            img_paths.append(img_path)
            labels.append(label)
    img_paths = np.array(img_paths)
    labels = np.array(labels)
    return img_paths, labels


def main():
    data_path = r'./cnt_revolve'
    save_dir = r'./data'
    img_paths, labels = get_img_label(data_path)
    for label in np.unique(labels):
        cls_dir = os.path.join(save_dir, label)
        if not os.path.exists(cls_dir):
            os.mkdir(cls_dir)
        cls_paths = img_paths[labels == label]
        for img_path in zip(cls_paths):
            shutil.copyfile(*img_path, os.path.join(cls_dir, os.path.split(*img_path)[-1]))


def train_val_main():
    data_dir = r'./data'
    cdir, dirs, files = next(os.walk(data_dir))
    for dir in dirs:
        if dir in ['train', 'val']:
            continue
        data_path = os.path.join(cdir, dir)
        img_paths, labels = get_img_label(data_path)
        x_train, y_train, _, _ = train_test_split(img_paths, labels, test_size=0.3, random_state=20, shuffle=True)

        # train
        mode_dir = os.path.join(cdir, 'train')
        if not os.path.exists(mode_dir):
            os.mkdir(mode_dir)
        if not os.path.exists(os.path.join(mode_dir, dir)):
            os.mkdir(os.path.join(mode_dir, dir))
        for path in x_train:
            name = os.path.split(path)[-1]
            shutil.copyfile(path, os.path.join(mode_dir, dir, name))

        # val
        mode_dir = os.path.join(cdir, 'val')
        if not os.path.exists(mode_dir):
            os.mkdir(mode_dir)
        if not os.path.exists(os.path.join(mode_dir, dir)):
            os.mkdir(os.path.join(mode_dir, dir))
        for path in y_train:
            name = os.path.split(path)[-1]
            shutil.copyfile(path, os.path.join(mode_dir, dir, name))



if __name__ == '__main__':
    train_val_main()
