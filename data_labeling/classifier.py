# -*- coding: utf-8 -*-
from transformers import BertModel
import torch
import torch.nn as nn
from utils.constants import c_len, BERT_BASE_UNCASED_PATH


class BertClassify(nn.Module):
    def __init__(self, args):
        super(BertClassify, self).__init__()
        bert_out = 768
        cnn_out = 768
        self.cnn = CNNClassifier(bert_out, args["num_filters"], (2, 3, 4), output_dim=cnn_out)
        self.cat = True
        self.cnn_norm = nn.LayerNorm(cnn_out)
        self.pool_norm = nn.LayerNorm(bert_out)
        if self.cat:
            self.linear = nn.Linear(bert_out+cnn_out, c_len)
        else:
            self.linear = nn.Linear(cnn_out, c_len)
        self.bert = BertModel.from_pretrained(BERT_BASE_UNCASED_PATH)
        self.dropout = nn.Dropout(args["dropout"])

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        seq_out, pooled_out = outputs[0], outputs[1]
        seq_out, pooled_out = self.dropout(seq_out), self.dropout(pooled_out)
        cnn_out = self.cnn(seq_out, attention_mask)
        if self.cat:
            # cnn_out = self.cnn_norm(cnn_out)
            # pooled_out = self.pool_norm(pooled_out)
            logits = self.linear(torch.cat([cnn_out, pooled_out], dim=1))
        else:
            logits = self.linear(cnn_out)
        logits = self.dropout(logits)
        return logits


class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_filters, ngram_filter_sizes=(2, 3), output_dim=None):
        super(CNNClassifier, self).__init__()
        self._ngram_filter_sizes = ngram_filter_sizes
        self._activation = nn.ReLU()
        self._output_dim = output_dim

        self._convolution_layers = [
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=num_filters,
                kernel_size=ngram_size) for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module('conv_layer_%d' % i, conv_layer)

        maxpool_output_dim = num_filters * len(self._ngram_filter_sizes)
        if self._output_dim:
            self.projection_layer = nn.Linear(maxpool_output_dim, self._output_dim)
        else:
            self.projection_layer = None
            self._output_dim = maxpool_output_dim

    def forward(self, tokens, mask=None):
        """
        Args:
            tokens (:class:`torch.FloatTensor` [batch_size, num_tokens, input_dim]): Sequence
                matrix to encoder.
            mask (:class:`torch.FloatTensor`): Broadcastable matrix to `tokens` used as a mask.
        Returns:
            (:class:`torch.FloatTensor` [batch_size, output_dim]): Encoding of sequence.
        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).float()
        tokens = torch.transpose(tokens, 1, 2)
        filter_outputs = []
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, 'conv_layer_{}'.format(i))
            filter_outputs.append(self._activation(convolution_layer(tokens)).max(dim=2)[0])
        # Now we have a list of `num_conv_layers` tensors of shape `(batch_size, num_filters)`.
        # Concatenating them gives us a tensor of shape
        # `(batch_size, num_filters * num_conv_layers)`.
        maxpool_output = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]
        if self.projection_layer:
            result = self.projection_layer(maxpool_output)
        else:
            result = maxpool_output
        return result
