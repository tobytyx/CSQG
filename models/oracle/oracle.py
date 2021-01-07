# -*- coding: utf-8 -*-
import torch as t
import torch.nn as nn
from torch.nn import functional as F
from models.join.rnn import forward_rnn
from models.join.attention import MLB_attention

num_classes = 3


class OracleNetwork(nn.Module):
    def __init__(self, opt, tokenizer):
        # 具有可学习参数的层放在构造函数中
        super(OracleNetwork, self).__init__()
        self.word_num = tokenizer.no_words
        self.word_embedding_dim = opt["embedding_dim"]
        self.category_num = opt["n_category"]
        self.category_embedding_dim = opt["category_embed_dim"]
        self.hidden_size = opt["hidden"]
        self.n_layers = opt["layer"]
        self.dropout = opt["dropout"]
        self.MLP_dim = opt["MLP_hidden"]
        self.attention = opt["attention"]  # 是否采取attention
        self.opt = opt
        self.lstm = nn.LSTM(self.word_embedding_dim, self.hidden_size, self.n_layers,
                          dropout=(0 if self.n_layers == 1 else self.dropout))
        self.word_embedding = nn.Embedding(self.word_num, self.word_embedding_dim)
        fc_dim = 0
        if not opt["crop"] and (not opt["image"]):
            fc_dim = self.hidden_size
        else:
            if opt["image"] and not opt["crop"]:
                self.linear_v1 = nn.Linear(opt["image_dim"], opt["fusion_dim"])  # W将图像信息映射到指定维度
            if opt["crop"] and not opt["image"]:
                self.linear_v2 = nn.Linear(opt["crop_dim"], opt["fusion_dim"])  # W将object信息映射到指定维度
            if opt["image"] and opt["crop"]:
                self.linear_v1 = nn.Linear(opt["image_dim"], opt["fusion_dim"])  # W将图像信息映射到指定维度
                self.linear_v2 = nn.Linear(opt["crop_dim"], opt["fusion_dim"])
                fc_dim += opt["fusion_dim"]
            self.linear_q = nn.Linear(self.hidden_size, opt["fusion_dim"])  # W将语句信息映射到指定维度
            fc_dim += opt["fusion_dim"]
        if opt["category"]:
            self.category_embedding = nn.Embedding(self.category_num + 1, self.category_embedding_dim)
            fc_dim += self.category_embedding_dim
        if opt["spatial"]:
            fc_dim += 8  # 空间位置维度信息
        if self.attention:
            self.atten = nn.Linear(opt["fusion_dim"], 1)
        self.fc1 = nn.Linear(fc_dim, self.MLP_dim)
        self.fc2 = nn.Linear(self.MLP_dim, num_classes)

    def forward(self, input):
        image, crop, category, spatial, question, lengths = input
        question = question.long()  # tensor.long()将类型转为Long
        word_embed = self.word_embedding(question)  # embedding传入的indices需要是Long类型的变量
        _, state = forward_rnn(word_embed, lengths, self.lstm, None, batch_first=True)
        hidden = state[0].squeeze_(0)
        if self.opt["crop"]:
            if self.attention and self.opt["crop_arch"].startswith('resnet'):
                # 只有Resnet可以实现attention
                # todo：修改数据处理方式，使得内存不够的情况下程序能够运行
                x = MLB_attention(image, hidden, self.linear_v1, self.linear_q, self.atten)
            else:
                hq = self.linear_q(hidden)
                if self.opt["crop_arch"].startswith('resnet'):
                    crop = F.avg_pool2d(crop, 7)
                    crop = crop.view(crop.size(0), -1)
                hc = self.linear_v2(crop)
                x = t.mul(hq, hc)
        if self.opt["image"]:
            if self.attention and self.opt["image_arch"].startswith('resnet'):
                x = MLB_attention(image, hidden, self.linear_v1, self.linear_q, self.atten)
            else:
                hq = self.linear_q(hidden)
                if self.opt["image_arch"].startswith('resnet'):
                    image = F.avg_pool2d(image, 7)
                    image = image.view(image.size(0), -1)
                hv = self.linear_v1(image)
                x = t.mul(hq, hv)
        if not self.opt["crop"] and (not self.opt["image"]):
            x = hidden
        if self.opt["category"]:
            category = category.long()
            cate = self.category_embedding(category)
            x = t.cat((x, cate), dim=1)
        if self.opt["spatial"]:
            spatial = spatial.float()
            x = t.cat((x, spatial), dim=1)
        y = nn.ReLU()(self.fc1(x))
        y = self.fc2(y)
        return y
