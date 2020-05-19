#!/usr/bin/env python

# -*- encoding: utf-8

'''
    _____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
    \__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
     /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
     \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
     / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
     \/               \/        \/               \/          \/       \/         \/     
 
 ==========================================================================================

@author: Yekun Chai

@license: School of Informatics, Edinburgh

@contact: chaiyekun@gmail.com

@file: Layers.py

@time: 29/09/2019 20:51 

@desc：       
               
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import math
import numpy as np

import utils


class LayerNorm(nn.Module):
    """ layer norm"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class SublayerConnection(nn.Module):
    """
    a residual connection followed by a layer norm
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ Apply residual connection to any sublayer with the same size"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """ encoder consists of a self-attn and ffc"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = utils.clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.local_rnn = LocalRNNLayer(size, dropout)

    def forward(self, x, mask):
        x = self.local_rnn(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """ decoder"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublyer = utils.clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublyer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublyer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublyer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    scaled dot product
    ---------------------------
    L : target sequence length
    S : source sequence length:
    N : batch size
    E : embedding dim

    h : # of attn head
    d_k: E // h
    ---------------------------
    :param query: (N, h, L, d_k)
    :param key: (N, h, S, d_k)
    :param value: (N, h, S, d_k)
    :param mask:
    :param dropout: float
    :return:
    """
    d_k = query.size(-1)
    # (nbatch, h, seq_len, d_k) @ (nbatch, h, d_k, seq_len) => (nbatch, h, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout:
        p_attn = dropout(p_attn)
    # (nbatch, h, seq_len, seq_len) * (nbatch, h, seq_len, d_k) = > (nbatch, h, seq_len, d_k)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        """
        multi-head attention
        :param h: nhead
        :param d_model: d_model
        :param dropout: float
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        #  assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = utils.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        ---------------------------
        L : target sequence length
        S : source sequence length:
        N : batch size
        E : embedding dim
        ---------------------------
        :param query: (N,L,E)
        :param key: (N,S,E)
        :param value: (N,S,E)
        :param mask:
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # batch size

        # 1) split embedding dim to h heads : from d_model => h * d_k
        # dim: (nbatch, h, seq_length, d_model//h)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) compute attention
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        # dim: (nbatch, h, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """ FFN """

    def __init__(self, d_model, d_ff, dropout=.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # return self.w_2(self.dropout(F.relu(self.w_1(x))))

        ## Swish
        x = self.w_1(x)
        x *= F.sigmoid(x)
        return self.w_2(self.dropout(x))


class LocalRNNLayer(nn.Module):
    def __init__(self, size, dropout=.0):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(size, size, window_size=5)
        self.sublayer = SublayerConnection(size, dropout)

    def forward(self, x):
        return self.sublayer(x, self.local_rnn)


class LocalRNN(nn.Module):
    """ R transformer"""

    def __init__(self, input_size, output_size, window_size, rnn_type='GRU', MAX_LENGTH=10000):
        super(LocalRNN, self).__init__()
        self.window_size = window_size
        if rnn_type == 'GRU':
            # set `batch_first`=True so that the input and output dim are both (nBatch, seq_len, d_model)
            self.rnn = nn.GRU(output_size, output_size, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_size, output_size, batch_first=True)
        else:
            self.rnn = nn.RNN(output_size, output_size, batch_first=True)
        # self.output = nn.Sequential(nn.Linear(output_size, output_size), nn.ReLU())

        # generate segments according to window_size.
        # -> e.g. window size = 4, generate [1,2,3,4,
        #                                      2,3,4,5,
        #                                        3,4,5,6,
        #                                          4,5,6,7,
        #                                              ...
        #                                                  MAX_LEN - 1 -k ,... , MAX_LEN-2, MAX_LEN-1]
        idx = [i for j in range(window_size - 1, MAX_LENGTH) for i in range(j - (window_size - 1), j + 1)]
        self.idx = torch.LongTensor(idx)
        # padding (k-1) before the beginning of the sequence
        self.zeros_pad = torch.zeros((window_size - 1, input_size))

    def forward(self, x):
        """ regard window size dim as batch dim"""
        assert x.dim() == 3, '3 dimensions of input expected!'
        nbatches, seq_len, d_model = x.size()

        x = self._gather_seg_sequence(x)
        output, _ = self.rnn(x)
        h_last_per_batch = output[:, -1, :]
        return h_last_per_batch.view(nbatches, seq_len, d_model)

    def _gather_seg_sequence(self, x):
        bsz, seq_len, d_model = x.size()
        # use `repeat` to pad one batch -> (bsz, k01, input_size)
        zeros = self.zeros_pad.repeat(bsz, 1, 1)
        # concat padded zeros and the sequence along the sequence dim
        x = torch.cat((zeros, x), dim=1)
        # gather the corresponding embeddings along the sequence dim (1)
        idx = self.idx[:self.window_size * seq_len]  #
        x_ = torch.index_select(input=x, dim=1, index=idx)
        # reshape -> (bsz * seq_len, window_size, d_model)
        x_ = x_.reshape(nbatch * seq_len, self.window_size, -1)
        return x_


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # to make positional encoding smaller


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * - (math.log(1e4) / d_model))
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention_RPR(nn.Module):
    """ @ author: Yekun CHAI """

    def __init__(self, d_model, h, max_relative_position, dropout=.0):
        """
        multi-head attention
        :param h: nhead
        :param d_model: d_model
        :param dropout: float
        """
        super(MultiHeadedAttention_RPR, self).__init__()
        assert d_model % h == 0
        #  assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = utils.clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

        self.max_relative_position = max_relative_position
        self.vocab_size = max_relative_position * 2 + 1
        self.embed_K = nn.Embedding(self.vocab_size, self.d_k)
        self.embed_V = nn.Embedding(self.vocab_size, self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        ---------------------------
        L : target sequence length
        S : source sequence length:
        N : batch size
        E : embedding dim
        ---------------------------
        :param query: (N,L,E)
        :param key: (N,S,E)
        :param value: (N,S,E)
        :param mask:
        """
        nbatches = query.size(0)  # batch size
        seq_len = query.size(1)
        # 1) split embedding dim to h heads : from d_model => h * d_k
        # dim: (nbatch, h, seq_length, d_model//h)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) rpr
        relation_keys = self.generate_relative_positions_embeddings(seq_len, seq_len, self.embed_K)
        relation_values = self.generate_relative_positions_embeddings(seq_len, seq_len, self.embed_V)
        logits = self._relative_attn_inner(query, key, relation_keys, True)
        weights = self.dropout(F.softmax(logits, -1))
        x = self._relative_attn_inner(weights, value, relation_values, False)
        # 3) "Concat" using a view and apply a final linear.
        # dim: (nbatch, h, d_model)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def _generate_relative_positions_matrix(self, len_q, len_k):
        """
        genetate rpr matrix
        ---------------------------
        :param len_q: seq_len
        :param len_k: seq_len
        :return: rpr matrix, dim: (len_q, len_q)
        """
        assert len_q == len_k
        range_vec_q = range_vec_k = torch.arange(len_q)
        distance_mat = range_vec_k.unsqueeze(0) - range_vec_q.unsqueeze(-1)
        disntance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        return disntance_mat_clipped + self.max_relative_position

    def generate_relative_positions_embeddings(self, len_q, len_k, embedding_table):
        """
        generate relative position embedding
        ----------------------
        :param len_q:
        :param len_k:
        :return: rpr embedding, dim: (len_q, len_q, d_k)
        """
        relative_position_matrix = self._generate_relative_positions_matrix(len_q, len_k)
        return embedding_table(relative_position_matrix)

    def _relative_attn_inner(self, x, y, z, transpose):
        """
        efficient implementation
        ------------------------
        :param x:
        :param y:
        :param z:
        :param transpose:
        :return:
        """
        nbatches = x.size(0)
        heads = x.size(1)
        seq_len = x.size(2)

        # (N, h, s, s)
        xy_matmul = torch.matmul(x, y.transpose(-1, -2) if transpose else y)
        # (s, N, h, d) => (s, N*h, d)
        x_t_v = x.permute(2, 0, 1, 3).contiguous().view(seq_len, nbatches * heads, -1)
        # (s, N*h, d) @ (s, d, s) => (s, N*h, s)
        x_tz_matmul = torch.matmul(x_t_v, z.transpose(-1, -2) if transpose else z)
        # (N, h, s, s)
        x_tz_matmul_v_t = x_tz_matmul.view(seq_len, nbatches, heads, -1).permute(1, 2, 0, 3)
        return xy_matmul + x_tz_matmul_v_t


if __name__ == '__main__':
    # import os
    #
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    #
    # plt.figure(figsize=(15, 5))
    # pe = PositionalEncoding(20, 0)
    # y = pe.forward(Variable(torch.zeros(1, 100, 20)))
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(["dim %d" % p for p in list(range(4, 8))])
    # plt.show()
    pe = MultiHeadedAttention_RPR(8, 256)
    x = torch.randn((64, 10, 256))
    y = pe(x, x, x)
    print(y.size())
