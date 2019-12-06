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

@file: Transformer_xl.py

@time: 17/10/2019 15:11 

@desc：       
               
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerXL(nn.Module):
    def __init__(self, d_model, n_head, d_head, mem_len, n_layer, clamp_len, tgt_len,
                 ext_len, dropatt, d_inner, pre_lnorm, dropout):
        self.n_layer = n_layer
        self.mem_len = mem_len
        self.word_emb = _
        self.clamp_len = clamp_len
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.drop = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers = self.layers.append(RelPartialLearnableDecLayer(
                n_head, d_model, d_head, d_inner, dropout, tgt_len=tgt_len,
                ext_len=ext_len, mem_len=mem_len, dropatt=dropatt, pre_lnorm=pre_lnorm))

    def _create_params(self):
        self.pos_emb = PositionEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for _ in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:  # do not use mems
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None: return

        assert len(hids) == len(mems), 'len(hids) != len(mems)!'

        with torch.no_grad():
            new_mems = []
            end_idx = mlen + qlen
            beggin_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat((mems[i], hids[i]), dim=0)
                new_mems.append(cat[beggin_idx:end_idx].detach())
        return new_mems

    def _forward(self, inp, mems=None):
        qlen, bsz = inp.szie()
        word_emb = self.word_emb(inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagnal=1 + mlen).byte()[:, :, None]

        hiddens = []
        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)

        hiddens.append(core_out)

        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hiddens.append(core_out)

        core_out = self.drop(core_out)

        new_mems = self._update_mems(hiddens, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, x, y, *mems):
        if not mems: mems = self.init_mems()

        tgt_len = y.size(0)
        hidden, new_mems = self._forward(x, mems=mems)
        pred_hid = hidden[-tgt_len:]


class RelPartialLearnableDecLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnableDecLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMHDPA(n_head, d_model, d_head, dropout, **kwargs)
        self.ffn = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        out = self.dec_attn(inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)
        out = self.ffn(out)
        return out


class RelPartialLearnableMHDPA(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, tgt_len=None, mem_len=None, pre_lnorm=False):
        super(RelPartialLearnableMHDPA, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.drop = nn.Dropout(p=dropout)
        self.dropatt = nn.Dropout(p=dropout)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** .5)
        self.pre_ln = pre_lnorm
        # xl
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x, zero_triu=False):
        bsz, klen, n_head, d_head = x.size()
        zero_pad = torch.zeros((bsz, 1, n_head, d_head), device=x.device, dtype=x.dtype)
        x_padded = torch.cat((zero_pad, x), 1)  # bsz, klen+1, n_head, d_head
        x_padded = x_padded.view(klen + 1, bsz, n_head, d_head)
        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat((mems, w), 0)
            if self.pre_ln:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_ln:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)  # qlen, bsz, n_head, d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)  # memlen + qlen, bsz, n_head, d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)  # memlen + qlen, bsz, n_head, d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum('ibnd, jbnd->ijbn', (rw_head_q, w_head_q))

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        # qlen, klen, bsz, n_head
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_mask.float().masked_fill(attn_mask[None, :, :, None], -float('inf')).type_as(
                    attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_mask.float().masked_fill(attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        attn_p = F.softmax(attn_score, -1)
        attn_p = self.dropatt(attn_p)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_p, w_head_v))

        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_ln:
            out = w + attn_out
        else:
            out = self.layer_norm(w + attn_out)
        return out


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_ln=False):
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.coreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_ln = pre_ln

    def forward(self, inp):
        core_out = self.coreNet(inp)
        if self.pre_ln:
            out = core_out + inp
        else:
            out = self.layer_norm(inp + core_out)
        return out


class PositionEmbedding(nn.Module):
    """ R_{i-j} in Att_rel in xl """

    def __init__(self, d_emb):
        super(PositionEmbedding, self).__init__()
        self.d_emb = d_emb
        inv_freq = 1 / (10000 ** (torch.arange(.0, d_emb, 2.0) / d_emb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinuisoid_inp = torch.ger(pos_seq, self.inv_freq)  # outer product

        if bsz is not None:
            return pos_seq[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_seq[:, None, :]
