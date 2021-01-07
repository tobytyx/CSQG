# -*- coding: utf-8 -*-
import os
import subprocess
import json
import numpy as np
import shutil
import spacy
import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import urllib
import logging
from logging.handlers import RotatingFileHandler


def execute(cmd, wait=True, printable=True):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if printable: print('[utils.execute] "%s"' % cmd)
    if wait:
        out, err = p.communicate()  # 等待程序运行，防止死锁
        out, err = out.decode('utf-8'), err.decode('utf-8')  # 从bytes转为str
        if err:
            raise ValueError(err)
        else:
            return out


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        print('[info] utils.ensure_dirname: removing dirname: %s' % os.path.abspath(dirname))
        shutil.rmtree(dirname)
    if not os.path.exists(dirname):
        print('[info] utils.ensure_dirname: making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname)


# 使得json.dumps可以支持ndarray，直接cls=utils.JsonCustomEncoder
class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def split_filepath(filename):
    absname = os.path.abspath(filename)
    dirname, basename = os.path.split(absname)
    split_tmp = basename.rsplit('.', maxsplit=1)
    if len(split_tmp) == 2:
        rootname, extname = split_tmp
    elif len(split_tmp) == 1:
        rootname = split_tmp[0]
        extname = None
    else:
        raise ValueError("programming error!")
    return dirname, rootname, extname


def download_file(fileurl, filedir=None, progress_bar=True, override=False, fast=False, printable=True):
    if filedir:
        ensure_dirname(filedir)
        assert os.path.isdir(filedir)
    else:
        filedir = ''
    filename = os.path.abspath(os.path.join(filedir, fileurl.split('/')[-1]))
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print("[info] utils.download_file: %s not exist, automatic makedir." % dirname)
    if not os.path.exists(filename) or override:
        if fast:
            p = subprocess.Popen('axel -n 10 -o {0} {1}'.format(filename, fileurl), shell=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(p.stdout.readline, ''):
                if line:
                    print(line.decode('utf-8').replace('\n', ''))
                else:
                    p.kill()
                    break
        else:
            if progress_bar:
                def my_hook(t):
                    last_b = [0]

                    def inner(b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            t.total = tsize
                        t.update((b - last_b[0]) * bsize)
                        last_b[0] = b

                    return inner

                with tqdm(unit='B', unit_scale=True, miniters=1,
                          desc=fileurl.split('/')[-1]) as t:
                    urllib.request.urlretrieve(fileurl, filename=filename,
                                               reporthook=my_hook(t), data=None)
            else:
                urllib.request.urlretrieve(fileurl, filename=filename)
        if printable: print("[info] utils.download_file: %s downloaded sucessfully." % filename)
    else:
        if printable: print("[info] utils.download_file: %s already existed" % filename)
    return filename


class MyConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, seed=None, p=None, af=None,
                 dim=None):
        super(MyConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.p = p  # 表示输入的dropout大小
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('[error] putils.Conv1d(%s, %s, %s, %s): input_dim (%s) should equal to 3' %
                             (self.in_channels, self.out_channels, self.kernel_size, self.stride, x.dim()))
        # x: b*49*512
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = x.transpose(1, 2)  # b*2048*36
        x = self.conv(x)  # b*310*36
        x = x.transpose(1, 2)  # b*36*310

        if self.af:
            if self.af == 'softmax':
                x = getattr(torch, self.af)(x, dim=self.dim)
            else:
                x = getattr(torch, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, seed=None, p=None, af=None, dim=None):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.af = af
        self.dim = dim
        if seed:
            torch.manual_seed(seed)
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        if x.size()[-1] != self.in_features:
            raise ValueError(
                '[error] putils.Linear(%s, %s): last dimension of input(%s) should equal to in_features(%s)' %
                (self.in_features, self.out_features, x.size(-1), self.in_features))
        if self.p:
            x = F.dropout(x, p=self.p, training=self.training)
        x = self.linear(x)
        if self.af:
            if self.af == 'softmax':
                x = getattr(torch, self.af)(x, dim=self.dim)
            else:
                x = getattr(torch, self.af)(x)
        return x

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MyATT(nn.Module):
    # 这个模块将obejct比较后的结果通过Conv1d转为attention map,并与v_feature_low结合得到最终的v_final
    def __init__(self, fuse_dim, glimpses, att_dim, seed=None, af='tanh'):
        super(MyATT, self).__init__()
        assert att_dim % glimpses == 0
        self.glimpses = glimpses
        self.att_dim = att_dim
        self.conv_att = MyConv1d(fuse_dim, glimpses, 1, 1, seed=seed, p=0.5, af='softmax', dim=1)
        self.af = af

    def forward(self, inputs, fuse):
        # v_final, alphas = self.att(v_feature, vq)
        # v_feature_low:b*32*310, vq:b*36*(36*310)
        x_att = self.conv_att(fuse)  # b*36*2, attention map

        tmp = torch.bmm(x_att.transpose(1, 2), inputs)  # b*2*310

        list_v_att = [e.squeeze() for e in torch.split(tmp, 1, dim=1)]  # b*310, b*310

        # list_att = torch.split(x_att, 1, dim=2)  # b*36, b*36

        x_v = torch.cat(list_v_att, -1)
        return x_v, x_att

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


class MyEmbedding(nn.Module):
    def __init__(self, vocab_list, dim=300, af='tanh'):
        super(MyEmbedding, self).__init__()
        self.vocab_list = vocab_list
        self.af = af
        self.embedding = nn.Embedding(
            num_embeddings=len(self.vocab_list),
            embedding_dim=dim,
            padding_idx=0
        )
        torch.nn.init.normal_(self.embedding.weight, 0.0, 0.1)

    def forward(self, input):
        embed = self.embedding(input)
        if self.af:
            if self.af == 'tanh':
                embed = torch.tanh(embed)
            elif self.af == 'relu':
                embed = torch.relu(embed)
        return embed


class Glove(nn.Module):
    def __init__(self, vocab_list, dim=300, af='tanh'):
        super(Glove, self).__init__()
        self.vocab_list = vocab_list
        self.af = af
        self.dim = dim

        self.nlp_glove = spacy.load('en_vectors_web_lg')
        self.glove_dict = []
        self.glove_embedding = nn.Embedding(num_embeddings=len(self.vocab_list),
                                            embedding_dim=300,
                                            padding_idx=0)
        self.load_glove_dict()
        if dim:
            self.embedding = nn.Embedding(num_embeddings=len(self.vocab_list),
                                          embedding_dim=dim,
                                          padding_idx=0)
            self.glove_embedding.weight.requires_grad = False
        else:
            self.glove_embedding.weight.requires_grad = True

    def load_glove_dict(self):
        # 计算小范围词典的表示
        glove_represents = []
        # 统计vocab中没有在glove词典中的单词
        count = 0
        for i, e in enumerate(self.vocab_list):
            if e not in self.nlp_glove.vocab:
                count += 1
            tmp = self.nlp_glove(u'%s' % e).vector
            if tmp.ndim == 2:
                tmp = tmp[0]
            glove_represents.append(tmp)

        print('[info] putils.Glove: %s/%s words not in Glove representation.'
              % (count, len(self.vocab_list)))
        glove_represents = torch.from_numpy(np.array(glove_represents))
        self.glove_embedding.load_state_dict({
            'weight': glove_represents})

    def forward(self, x):
        embed = self.glove_embedding(x)
        if self.dim:
            raw_embed = self.embedding(x)
            if self.af:
                if self.af == 'tanh':
                    raw_embed = torch.tanh(raw_embed)
                elif self.af == 'relu':
                    raw_embed = torch.relu(raw_embed)
            embed = torch.cat((raw_embed, embed), dim=-1)

        return embed

    def __call__(self, *input, **kwargs):
        return super().__call__(*input, **kwargs)


def create_logger(save_path, mode):
    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path, mode, 0, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.INFO)
    logger.addHandler(steam_handler)

    return logger
