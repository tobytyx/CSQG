# -*- coding: utf-8 -*-
# import numpy as np
import torchvision.transforms as transforms


def get_spatial_feat(bbox, im_width, im_height):
    # normalize image height and width to be 2, and place the origin at the image center,
    # so that coordinates range from -1 to 1.
    # return [x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box]

    x_min = (bbox.x / im_width) * 2 - 1
    x_max = ((bbox.x + bbox.width) / im_width) * 2 - 1
    x_center = ((bbox.x + 0.5*bbox.width) / im_width) * 2 - 1

    y_min = (bbox.y / im_height) * 2 - 1
    y_max = ((bbox.y + bbox.height) / im_height) * 2 - 1
    y_center = ((bbox.y + 0.5*bbox.height) / im_height) * 2 - 1

    w_box = (bbox.width / im_width) * 2
    h_box = (bbox.height / im_height) * 2

    # Concatenate features
    feat = [x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box]
    # feat = np.array(feat)

    return feat


def crop_object(bbox, raw_img):
    # Need to use integer only
    left = int(bbox.x)
    right = int(bbox.x + bbox.width)
    upper = int(bbox.y)
    lower = int(bbox.y + bbox.height)

    # Create crop with tuple defining by left, upper, right, lower (Beware -> y descending!)
    crop = raw_img.crop(box=(left, upper, right, lower))
    return crop


def get_transform(shape):
    # 为后面cnn网络提取特征，进行normalize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transfrom = transforms.Compose([
                                    transforms.Resize(shape),
                                    transforms.ToTensor(),
                                    normalize,
                                   ])
    return transfrom
