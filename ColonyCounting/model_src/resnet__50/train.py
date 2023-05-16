# coding

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import f1_score
from torchvision.models import resnet50, vgg16, vgg19,resnet152, wide_resnet50_2
from torch.utils.tensorboard import SummaryWriter


img_path = './data'
frozen_blocks = 0
weights = torch.Tensor([4.468, 1.0, 6.872, 24.343, 94.096, 257.526, 305.812, 699.0, 699.0, 543.667])
lr = 0.0001
writer = SummaryWriter(log_dir='./logs')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])}

    data_root = os.path.abspath(img_path)  # get data root path
    image_path = data_root  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # net = resnet50().to(device)
    net = wide_resnet50_2(pretrained=True)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet_50-333f7ec4.pth
    # model_weight_path = "resnet_50-pre.pth"
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for i, child in enumerate(net.children()):
    #     if i >= frozen_blocks:
    #         continue
    #     for param in child.parameters():
    #         param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, len(cla_dict))
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss(weight=weights.to(device))

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    epochs = 200
    best_micro = 0.0
    best_macro = 0.0
    save_path = 'resnet_50.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)



        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        micro_f1 = 0.0
        macro_f1 = 0.0
        cnt =0
        val_steps = len(validate_loader)
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                cnt += 1
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                predict_y = np.array((predict_y.to('cpu')))
                micro_f1 += f1_score(val_labels, predict_y, average='micro')
                macro_f1 += f1_score(val_labels, predict_y, average='macro')
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        micro_f1 /= val_steps
        macro_f1 /= val_steps
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  val_micro_f1: %.3f val_macro_f1: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate, micro_f1, macro_f1))

        writer.add_scalar('train loss', scalar_value=running_loss / train_steps, global_step=epoch)
        writer.add_scalar('acc', scalar_value=val_accurate, global_step=epoch)
        writer.add_scalar('micro f1', scalar_value=micro_f1, global_step=epoch)
        writer.add_scalar('macro f1', scalar_value=macro_f1, global_step=epoch)

        if micro_f1 >= best_micro and macro_f1 >= best_macro:
            best_micro = micro_f1
            best_macro = macro_f1
            torch.save(net.state_dict(), save_path)

    writer.close()

    print('Finished Training')


if __name__ == '__main__':
    main()
