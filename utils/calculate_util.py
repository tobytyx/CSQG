# -*- coding: utf-8 -*-
import torch
import math
import numpy as np


def calculate_accuracy(pred, target):
    """
    计算正确率
    :param pred:  [B, C]
    :param target: [B]
    :return: 正确数目和正确率
    """
    predicted = torch.argmax(pred, dim=1)
    total = target.size(0)  # 总数
    correct = torch.sum(predicted == target).item()
    return correct, float(correct/total)


def attention_crossentropy(attention):
    # 传入36个attention分布，计算注意力的交叉熵
    cross = 0
    for i in range(36):
        if isinstance(attention[i], torch.Tensor):
            p = attention[i].detach().cpu().numpy()
        else:
            p = attention[i]
        n = p*math.log(p)
        cross -= n
    return cross


def european_distance(vec1, vec2):
    # 计算两个向量的欧式距离
    return np.linalg.norm(vec1 - vec2)


def cosine_distance(vec1, vec2):
    # 计算两个向量的余弦距离和余弦相似度
    num = float(np.sum(vec1 * vec2))
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return cos, sim