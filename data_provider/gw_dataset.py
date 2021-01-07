# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import os
import pickle


class GWDataset(Dataset):
    def __init__(self, data_dir, dataset, config, tokenizer):
        self.tokenizer = tokenizer
        multi_cate = config["multi_cate"] if "multi_cate" in config else False
        if dataset is not None:
            self._load_data_from_dataset(data_dir, dataset, multi_cate=multi_cate)
        self.config = config
        self.img_provider = None
        feature_dir = os.path.join(data_dir, "features")
        if "image" in config and config["image"]:
            arch = config["image_arch"]
            feature = config["image_feature"]
            self.image_builder = ImageProvider(
                os.path.join(feature_dir, arch, feature, "image.pkl"), "feature", "vgg16")

        if "crop" in config and config["crop"]:
            arch = config["crop_arch"]
            feature = config["crop_feature"]
            self.crop_builder = ImageProvider(
                os.path.join(feature_dir, arch, feature, "crop.pkl"), "feature", "vgg16")

        if "object" in config and config["object"]:
            arch = "rcnn"
            self.object_builder = ImageProvider(
                os.path.join(feature_dir, arch, "size,rcnn_arch,224.txt"), "file", "rcnn")

    def _load_data_from_dataset(self, data_dir, dataset, state_filter=("success",), multi_cate=False):
        raise NotImplementedError

    def load_data_from_games(self, games):
        # infer from games
        self.games = games

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        raise NotImplementedError


class ImageProvider(object):
    def __init__(self, path, mode, arch):
        self.mode = mode
        self.arch = arch
        self.k2v = {}
        if mode == "file":
            fea_dir = os.path.dirname(path)
            with open(path, mode="r") as f:
                for filename in f.readlines():
                    name = os.path.basename(filename.strip())
                    self.k2v[name] = os.path.join(fea_dir, "size,rcnn_arch,224,att_pos", name[:-3] + "pkl")
        elif mode == "feature":
            with open(path, mode="rb") as f:
                self.k2v = pickle.load(f)
        else:
            raise

    def load_feature(self, key):
        value = self.k2v[key]
        if self.mode == "file":
            with open(value, mode="rb") as f:
                value = pickle.load(f)
        return value
