import os
import time

import cv2
import torch
from torchvision.transforms import transforms

from .src import u2net_full


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(model, origin_img, transform=None):
    if isinstance(origin_img, str):
        assert os.path.exists(origin_img), f"image file {origin_img} dose not exists."
        origin_img = cv2.cvtColor(cv2.imread(origin_img, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    h, w = origin_img.shape[:2]

    if transform:
        img = transform(origin_img)
    else:
        img = origin_img.copy()
        img = transforms.ToTensor()(img)
    img = torch.unsqueeze(img, 0).to(device)  # [C, H, W] -> [1, C, H, W]

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
        return origin_img, pred
        # pred_mask = np.where(pred > threshold, 1, 0)
        # origin_img = np.array(origin_img, dtype=np.uint8)
        # seg_img = origin_img * pred_mask[..., None]
        # plt.imshow(seg_img)
        # plt.show()
        # cv2.imwrite("pred_result.png", cv2.cvtColor(seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
