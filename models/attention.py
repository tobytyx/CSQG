# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class ConcatFusion(nn.Module):
    def __init__(self, q_dim, v_dim, hidden_dim):
        super(ConcatFusion, self).__init__()
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(q_dim+v_dim, hidden_dim, bias=True)
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.linear.bias, 0.0)
        self.drop = nn.Dropout(0.1)

    def forward(self, q, v):
        assert q.dim() == v.dim()
        fusion = torch.cat((q, v), dim=-1)
        fusion = self.linear(fusion)
        fusion = torch.relu(fusion)
        fusion = self.drop(fusion)
        return fusion


class VisualAttention(nn.Module):
    def __init__(self, v_feature, q_feature, mid_feature, glimpse, drop=0.1):
        super(VisualAttention, self).__init__()

        self.q_line = nn.Linear(q_feature, mid_feature, bias=True)
        torch.nn.init.kaiming_normal_(self.q_line.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.q_line.bias, 0.0)

        self.x_line = nn.Linear(mid_feature, glimpse)
        torch.nn.init.kaiming_normal_(self.x_line.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.x_line.bias, 0.0)

        self.v_conv = nn.Conv1d(v_feature, mid_feature, 1)
        torch.nn.init.kaiming_normal_(self.v_conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.v_conv.bias, 0.0)

        self.drop = nn.Dropout(drop)
        self.v_dim = v_feature
        self.q_dim = q_feature

    def forward(self, v, q):
        """
        Conv attention
        :param v: B * num * v_features
        :param q: B * q_features
        :return score: B * num
        """
        # ic(v.size(), q.size(), self.v_dim, self.q_dim)
        v = self.v_conv(v.transpose(1, 2))  # B * mid * num
        v = torch.relu(v.transpose(1, 2))  # B * num * mid
        q = self.q_line(q)  # B * mid
        q = q.unsqueeze(1).expand_as(v)  # B * num * mid
        x = torch.relu(q + v)
        x = self.x_line(self.drop(x))  # B * num * g
        x = torch.sum(x, dim=-1)  # B * num
        score = F.softmax(x, dim=-1)
        return score


class SessionAttention(nn.Module):
    def __init__(self, decoder_hidden, encoder_hidden, hidden_size):
        super().__init__()

        self.decoder_fc = nn.Linear(decoder_hidden, hidden_size)
        torch.nn.init.kaiming_normal_(self.decoder_fc.weight, nonlinearity='relu')
        self.encoder_fc = nn.Linear(encoder_hidden, hidden_size)
        torch.nn.init.kaiming_normal_(self.encoder_fc.weight, nonlinearity='relu')
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs, session_mask):
        """
        计算attention值
        :param hidden: B x decoder_hidden
        :param encoder_outputs: B x len x session_hidden
        :param session_mask:
        :return attention: B x len
        """
        hidden = self.decoder_fc(hidden)
        encoder_outputs = self.encoder_fc(encoder_outputs)  # B x len x hidden
        attn_energies = torch.einsum("bh,blh->bl", [hidden, encoder_outputs])
        attn_energies = attn_energies.masked_fill(session_mask, -np.inf)
        scores = F.softmax(attn_energies, dim=-1)
        return scores


class ConvAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(ConvAttention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):  # B x 1000, B x 600
        v = self.v_conv(self.drop(v))  # b*520*36
        q = self.q_lin(self.drop(q))  # b*520
        q = self.tile_2d_over_nd(q, v)
        x = self.relu(v + q)
        x = self.x_conv(self.drop(x))   # b*g*36*1
        return x

    @staticmethod
    def tile_2d_over_nd(feature_vector, feature_map):
        """ Repeat the same feature vector over all spatial positions of a given feature map.
            The feature vector should have the same batch size and number of features as the feature map.
        """
        n, c = feature_vector.size()
        spatial_size = feature_map.dim() - 2
        tiled = feature_vector.view(n, c, *([1] * spatial_size)).expand_as(feature_map)
        return tiled
