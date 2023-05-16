import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from torchvision.models import resnet50


img_path = r"C:\work\JunLuoCount-master\model_src\renet__50\data\val\8\3bd1c8e6762a27d22afda5bddd9a648b_26_8.png"
weights_path = "resnet_50_100epo.pth"
json_path = r'F:\final_version\final_version\JunLuoCounting\model_src\resnet__50\class_indices.json'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


def main(model, img, transform=None):


    if transform is None:
        transform = transforms.ToTensor()
    # [N, C, H, W]
    img_ = transform(img)
    # expand batch dimension
    img_ = torch.unsqueeze(img_, dim=0)

    # read class_indict
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img_.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # plt.show()

    return class_indict[str(predict_cla)]


if __name__ == '__main__':
    main()
