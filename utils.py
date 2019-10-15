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

@file: utils.py

@time: 29/09/2019 20:41 

@desc：       
               
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import numpy as np
import copy


def clones(module, N):
    """
    produce N identical layers
    :param module:
    :param N:
    :return:
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """ Mask out subsequent positions """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), .0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


if __name__ == '__main__':
    # test subsequent_mask
    # ---------------------------------
    # plt.figure(figsize=(5, 5))
    # plt.imshow(subsequent_mask(20)[0])
    # plt.show()
    # ---------------------------------
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # test label smoothing
    # crit = LabelSmoothing(5, 0, .4)
    # predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0], [0, 0.2, 0.7, 0.1, 0]])
    # v = crit(Variable(predict.log()),
    #          Variable(torch.LongTensor([2, 1, 0])))
    # plt.imshow(crit.true_dist)
    # plt.show()
    #
    crit = LabelSmoothing(5, 0, 0.1)

    def loss(x):
        d = x + 3 * 1
        predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
                                     ])
        # print(predict)
        return crit(Variable(predict.log()),
                    Variable(torch.LongTensor([1]))).data.item()


    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()
