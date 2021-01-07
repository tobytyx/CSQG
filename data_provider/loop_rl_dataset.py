# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from data_provider.gw_dataset import ImageProvider
from process_data.data_preprocess import get_games


class LoopRlDataset(Dataset):
    def __init__(self, data_dir, dataset, qgen_args, oracle_args, guesser_args, tokenizer):
        self.tokenizer = tokenizer
        self.qgen_args = qgen_args
        self.oracle_args = oracle_args
        self.guesser_args = guesser_args
        feature_dir = os.path.join(data_dir, "features")

        self.image_builder = ImageProvider(
            os.path.join(feature_dir, "vgg16", "fc8", "image.pkl"), "feature", "vgg16")

        self.crop_builder = ImageProvider(
            os.path.join(feature_dir, "vgg16", "fc8", "crop.pkl"), "feature", "vgg16")

        self.object_builder = ImageProvider(
            os.path.join(feature_dir, "rcnn", "size,rcnn_arch,224.txt"), "file", "rcnn")

        old_games = get_games(data_dir, dataset, True)
        self.games = [g for g in old_games if g.status == "success"]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, item):
        game = self.games[item]
        game_id = game.id
        if self.qgen_args["object"]:
            img = self.object_builder.load_feature(game.img.filename)
            qgen_img = img["att"]
            qgen_bbox = get_bbox(img["pos"])
        else:  # image
            qgen_img = self.image_builder.load_feature(game.img.id)
            qgen_bbox = np.random.randn(36, 76)

        oracle_img, oracle_crop = 0, 0
        if self.oracle_args["image"]:
            oracle_img = self.image_builder.load_feature(game.img.id)
        if self.oracle_args["crop"]:
            oracle_crop = self.crop_builder.load_feature(game.object.id)
        category = game.object.category_id
        spatial = game.object.spatial

        guesser_img = 0
        if self.guesser_args["image"]:
            guesser_img = self.image_builder.load_feature(game.img.id)
        cats = []
        spas = []
        target = 0
        for i, o in enumerate(game.objects):
            cats.append(o.category_id)
            # N x [x_min, y_min, x_max, y_max, x_center, y_center, w_box, h_box]
            spas.append(o.spatial)
            if o.id == game.object_id:
                target = i  # 目标object的index
        return qgen_img, qgen_bbox, oracle_img, oracle_crop, category, spatial, guesser_img, cats, spas, target, game_id


def loop_rl_collate(batch):
    q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_cats, g_spas, targets, game_ids = zip(*batch)

    q_imgs = torch.tensor(q_imgs, dtype=torch.float)
    q_bbox = torch.tensor(q_bbox, dtype=torch.float)
    o_imgs = torch.tensor(o_imgs, dtype=torch.float)
    o_crops = torch.tensor(o_crops, dtype=torch.float)
    o_cats = torch.tensor(o_cats, dtype=torch.long)
    o_spas = torch.tensor(o_spas, dtype=torch.float)
    g_imgs = torch.tensor(g_imgs, dtype=torch.float)
    g_obj_lens = [len(obj) for obj in g_cats]
    max_obj_lens = max(g_obj_lens)
    g_cats = [cat + (max_obj_lens - g_obj_lens[i]) * [0] for i, cat in enumerate(g_cats)]
    g_cats = torch.tensor(g_cats, dtype=torch.long)
    g_spas = [spa + (max_obj_lens - g_obj_lens[i]) * [[0] * 8] for i, spa in enumerate(g_spas)]
    g_spas = torch.tensor(g_spas, dtype=torch.float)
    g_obj_mask = [[1] * obj_len + (max_obj_lens - obj_len) * [0] for obj_len in g_obj_lens]
    g_obj_mask = torch.tensor(g_obj_mask, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    batch = [game_ids, q_imgs, q_bbox, o_imgs, o_crops, o_cats, o_spas, g_imgs, g_obj_mask, g_cats, g_spas, targets]
    return batch


def get_bbox(bbox):
    """
    :param bbox: numpy, 32 * [x_min, y_min, x_max, y_max]
    :return: numpy, 32 * [x, y, w, h, relative_x0, ..., relative_x35, relative_y0, relative_y35](76)
    """
    mu = np.mean(bbox, axis=0)
    sigma = np.std(bbox, axis=0)
    bbox = (bbox-mu) / sigma
    w = bbox[:, 2] - bbox[:, 0]
    w = w.reshape(w.shape[0], 1)
    h = bbox[:, 3] - bbox[:, 1]
    h = h.reshape(h.shape[0], 1)
    pos_x = (bbox[:, 2] + bbox[:, 0]) / 2
    pos_y = (bbox[:, 3] + bbox[:, 1]) / 2
    rel_x = np.array([pos_x[i] - pos_x for i in range(bbox.shape[0])])
    rel_y = np.array([pos_y[i] - pos_y for i in range(bbox.shape[0])])
    pos_x = pos_x.reshape(pos_x.shape[0], 1)
    pos_y = pos_y.reshape(pos_y.shape[0], 1)
    return np.concatenate([pos_x, pos_y, w, h, rel_x, rel_y], axis=1)