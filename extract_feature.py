import os
import argparse
from process_data.image_feature import extract_image_feature, extract_object_feature


parser = argparse.ArgumentParser(
    description='Extract image/object features',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
##################################################

parser.add_argument('--data_dir', default='../data/', type=str,
                    help='path to data dir')
parser.add_argument('--image_dir', default='../data/imgs/', type=str,
                    help='path to COCO images dir')
parser.add_argument('--fea_dir', default='../data/features/', type=str,
                    help='path to COCO images')
parser.add_argument('--type', default='image', type=str, choices=['image', 'crop'],
                    help='extract image or crop features')
parser.add_argument('--arch', default='vgg16', type=str, choices=["vgg16", "resnet152"],
                    help='convnet used to extract image feature')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size for every time extract feature')
parser.add_argument('--image_size', default=224, type=int,
                    help='resize image to fixed shape')
parser.add_argument('--feature', default='fc8', type=str, choices=['fc7', 'fc8', 'layer4', 'avgpool', 'fc'],
                    help='vgg layer')
parser.add_argument('--shape', default=(1000,), type=tuple,
                    help='extracted feature shape for image')


if __name__ == "__main__":
    flags = parser.parse_args()
    if not os.path.exists(flags.fea_dir):
        os.mkdir(flags.fea_dir)
    if flags.type == "image":
        extract_image_feature(flags)
    elif flags.type == "crop":
        extract_object_feature(flags)
