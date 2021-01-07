# -*- coding: utf-8 -*-
import argparse


def oracle_arguments():
    parser = argparse.ArgumentParser()
    trainArgs = parser.add_argument_group("train options")
    trainArgs.add_argument("--option", choices=["train", "test"], default="train")
    trainArgs.add_argument("--name", type=str, default="baseline")
    trainArgs.add_argument("--batch_size", type=int, default=128)
    trainArgs.add_argument("--display_step", type=int, default=2000)
    trainArgs.add_argument("--lr", type=float, default=1e-4)
    trainArgs.add_argument("--clip_val", type=int, default=5)
    trainArgs.add_argument("--epoch_num", type=int, default=15)
    trainArgs.add_argument("--early_stop", type=int, default=-1)

    modelArgs = parser.add_argument_group("model options")
    modelArgs.add_argument("--model", type=str, default="baseline")
    modelArgs.add_argument("--embedding_dim", type=int, default=300, help="word embedding dim")
    modelArgs.add_argument("--bidirectional", action="store_true", help="whether bi-directional")
    modelArgs.add_argument("--hidden", type=int, default=2400, help="encoder LSTM's hidden size")
    modelArgs.add_argument("--layer", type=int,  default=1, help="the layer of encoder LSTM")
    modelArgs.add_argument("--dropout", type=float, default=0, help="the dropout of encoder cell")
    modelArgs.add_argument("--n_category", type=int, default=90)
    modelArgs.add_argument("--category_embed_dim", type=int, default=512)
    modelArgs.add_argument("--MLP_hidden", type=int, default=512)
    modelArgs.add_argument("--image_arch", type=str, default="vgg16", help="the arch for image net")
    modelArgs.add_argument("--image_feature", type=str, default="fc8", help="the feature map of image")
    modelArgs.add_argument("--image_dim", type=int, default=1000, help="the dim of image")
    modelArgs.add_argument("--crop_arch", type=str, default="vgg16", help="the arch for image net")
    modelArgs.add_argument("--crop_feature", type=str, default="fc8", help="the feature map of image")
    modelArgs.add_argument("--crop_dim", type=int, default=1000, help="the dim of image")
    modelArgs.add_argument("--fusion_dim", type=int, default=512, help="fusion dim for question and image")
    modelArgs.add_argument("--attention", type=bool, default=False, help="use attention when fusion")

    inputArgs = parser.add_argument_group("input options")
    inputArgs.add_argument("--question", type=bool, default=True)
    inputArgs.add_argument("--category", type=bool, default=True)
    inputArgs.add_argument("--spatial", type=bool, default=True)
    inputArgs.add_argument("--image", type=bool, default=False)
    inputArgs.add_argument("--crop", type=bool, default=True)

    return parser