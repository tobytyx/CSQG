# -*- coding: utf-8 -*-
import argparse


def qgen_arguments():
    parser = argparse.ArgumentParser()
    trainArgs = parser.add_argument_group("train options")
    trainArgs.add_argument("--option", choices=["train", "test"], default="train")
    trainArgs.add_argument("--name", type=str, default="baseline")
    trainArgs.add_argument("--display_step", type=int, default=2000)
    trainArgs.add_argument("--batch_size", type=int, default=64)
    trainArgs.add_argument("--lr", type=float, default=1e-4)
    trainArgs.add_argument("--clip_val", type=int, default=5)
    trainArgs.add_argument("--epoch_num", type=int, default=50)
    trainArgs.add_argument("--early_stop", type=int, default=5)
    trainArgs.add_argument("--th", type=float, default=0.4)

    modelArgs = parser.add_argument_group("model options")
    modelArgs.add_argument("--model", type=str, default="baseline",
                           choices=["cat", "baseline", "cat_base", "hrnn", "cat_accu", "cat_attn"])
    modelArgs.add_argument("--multi_cate", default=False, action="store_true")
    modelArgs.add_argument("--embedding_dim", type=int, default=500, help="word embedding dim")
    modelArgs.add_argument("--category_dim", type=int, default=10, help="word embedding dim")
    modelArgs.add_argument("--answer_dim", type=int, default=10, help="word embedding dim")
    # model structure
    modelArgs.add_argument("--combine_type", type=str, default="embed", choices=["embed", "concat"],
                           help="concatenate or sum the answer and category info.")
    modelArgs.add_argument("--task", type=str, default="gen", choices=["cls", "gen", "cls_gen"],
                           help="predict category / generate tokens / both")
    modelArgs.add_argument("--no_gt_cate", default=False, action="store_true",
                           help="use predict category to train decoder")
    modelArgs.add_argument("--prior_weight_type", default="punish", choices=["punish", "prior"],
                           help="the way for model qgen_accu.")
    # query RNN
    modelArgs.add_argument("--query_hidden", type=int, default=800, help="first RNN encoder's hidden size")
    modelArgs.add_argument("--query_layer", type=int, default=1, help="the layer of first RNN")
    modelArgs.add_argument("--query_dropout", type=float, default=0, help="the dropout of query encoder cell")
    modelArgs.add_argument("--query_bi", action="store_true", help="whether bi-direction")
    # session RNN
    modelArgs.add_argument("--session_hidden", type=int, default=1000, help="session rnn's hidden size")
    modelArgs.add_argument("--session_layer", type=int, default=1, help="the layer of session RNN")
    modelArgs.add_argument("--session_dropout", type=int, default=0, help="the layer of session RNN")
    modelArgs.add_argument("--session_bi", action="store_true", help="whether bi-direction")
    # decoder RNN
    modelArgs.add_argument("--decoder_att", default=False, action="store_true")
    modelArgs.add_argument("--category_once", default=False, action="store_true")
    modelArgs.add_argument("--decoder_hidden", type=int, default=600, help="decoder LSTM's hidden size")
    modelArgs.add_argument("--decoder_layer", type=int, default=1, help="the layer of decode GRU")
    modelArgs.add_argument("--decoder_dropout", type=float, default=0, help="the dropout of decode cell")
    modelArgs.add_argument("--decoder_bi", action="store_true", help="whether bi-direction")
    modelArgs.add_argument("--decoder_attention_dim", type=int, default=800, help="the decoder attention dim")
    modelArgs.add_argument("--beam_size", type=int, default=2, help="the size of each beam-search")
    modelArgs.add_argument("--max_dialog_length", type=int, default=10)
    # classify
    modelArgs.add_argument("--concat_hidden", type=int, default=1000)

    # visual process
    modelArgs.add_argument("--image_arch", type=str, default="vgg16", help="the arch for image net")
    modelArgs.add_argument("--image_dim", type=int, default=1000, help="the dim of image")
    modelArgs.add_argument("--image_feature", type=str, default="fc8", help="the feature map of image")
    modelArgs.add_argument("--v_feature", type=int, default=520, help="the dim of visual feature")
    modelArgs.add_argument("--visual_att", default=False, action="store_true")
    modelArgs.add_argument("--visual_att_dim", type=int, default=800, help="the dim of attention")
    modelArgs.add_argument("--visual_dropout", type=int, default=0.2, help="visual dropout after compress")
    modelArgs.add_argument("--glimpse", type=int, default=1, help="the glimpse of attention")
    inputArgs = parser.add_argument_group("input options")
    inputArgs.add_argument("--image", default=False, action="store_true")
    inputArgs.add_argument("--crop", default=False, action="store_true")
    inputArgs.add_argument("--object", default=False, action="store_true")
    return parser
