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

@time: 17/10/2019 15:11 

@desc：       
               
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, d_embed):
        super(PositionalEmbedding, self).__init__()
        self.d_embed = d_embed
        inv_freq = 1 / (10000 ** (torch.arange(.0, d_embed, 2.0)) / d_embed)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, nbatch=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_embed = torch.concat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        if nbatch is None:
            return pos_embed[:, None, :]
        else:
            return pos_embed[:, None, :].expand(-1, nbatch, -1)


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_hid, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_hid
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(d_model, d_hid), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, x):
        if self.pre_lnorm:
            out = self.net(self.layer_norm(x))
            out += x
        else:
            out = self.net(x)
            out = self.layer_norm(x + out)
        return out


class MultiHeadAttn(nn.Module):
    def __init__(self, nheads, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()
        self.nheads = nheads
        self.d_model = d_model
        self.d_head = d_head  # d_model//nheads
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, nheads * d_head, d_model, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * nheads * d_head, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropatt)
        self.o_net = nn.Linear(nheads * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** .5)
        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        # [seq_len x nbatch x nheads x d_head]
        if mems is None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.nheads, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.nheads, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.nheads, self.d_head)

        # [q_len, k_len, nbatch, n_head]
        attn_score = torch.einsum('qbnd,kbnd->qkbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.mask_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.mask_fill_(attn_mask[:, :, :, None], -float('inf'))

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout_attn(attn_prob)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.nheads * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

        if self.pre_lnorm:
            output = h + attn_out
        else:
            output = self.layer_norm(h + attn_out)
        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, nheads, d_model, d_head, dropout_p, dropout_attn_p=0, tgt_len=None, ext_len=None, mem_len=None,
                 pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.nheads = nheads
        self.d_model = d_model
        self.d_head = d_head
        self.dropout_p = dropout_p
        self.qkv_net = nn.Linear(d_model, 3 * nheads * d_head, bias=False)

        self.dropout = nn.Dropout(dropout_p)
        self.dropout_attn = nn.Dropout(dropout_attn_p)
        self.o_net = nn.Linear(nheads * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)
        self.scale = 1 / (d_head ** .5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)), device=x.device, dtype=x.type)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_selected(mask[:, :, None, None]).view(qlen, klen, x.zie(2), x.size(3))
        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.concat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMulltiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMulltiHeadAttn, self).__init__(*args, **kwargs)
        self.r_net = nn.Linear(self.d_model, self.nheads * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, nbatch = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        klen = w_head_k.size(0)
        w_head_q = w_head_q.view(qlen, nbatch, self.nheads, self.d_head)
        w_head_k = w_head_k.view(klen, nbatch, self.nheads, self.d_head)
        w_head_v = w_head_v.view(klen, nbatch, self.nheads, self.d_head)

        r_head_k = r_head_k.view(rlen, self.nheads, self.d_head)

        # compute attn score
        rw_head_q = w_head_q + r_w_bias
        AC = torch.einsum('ibnd, jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_w_bias
        BD = torch.einsum('ibnd, jbnd->ijbn', (rr_head_q, r_head_k))
        BD = self._rel_shift(BD)

        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attn probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().mask_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().mask_fill(attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout_attn(attn_prob)

        # compute attn vec
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        attn_vec = attn_vec.view(attn_vec.size(0), attn_vec.size(1), self.nheads * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.pre_lnorm(w + attn_out)
        return output


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **Kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **Kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        qlen, nbatch = w.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, nbatch, self.nheads, self.d_head)
        w_head_k = w_head_k.view(klen, nbatch, self.nheads, self.d_head)
        w_head_v = w_head_v.view(klen, nbatch, self.nheads, self.d_head)

        if klen < r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_emb.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute the attn score
        rw_head_q = w_head_q + r_w_bias[None]

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))
        D_ = r_bias[None, :, :, None]
        BD = self._rel_shift(B_ + D_)

        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attn prob
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropout_attn(attn_prob)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        attn_vec = attn_vec.contiguous().view(attn_vec.size(0), attn_vec.size(1), self.nheads * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.dropout(attn_out)

        if self.pre_lnorm:
            output = w + attn_out
        else:
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, nheads, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttn(nheads, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, nheads, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()
        self.dec_attn = RelLearnableMultiHeadAttn(nheads, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r_emb, r_w_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class RelPartialLearnabledecoderLayer(nn.Module):
    def __init__(self, nheads, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelPartialLearnabledecoderLayer, self).__init__()
        self.dec_attn = RelPartialLearnableMulltiHeadAttn(nheads, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_input, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_input, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        output = self.pos_ff(output)
        return output


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** .5
        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()

        if div_val == 1:
            self.embed_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, input):
        if self.div_val == 1:
            embed = self.emb_layers[0][input]
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
            else:
                param = next(self.parameters())
                input_flat = input.view(-1)
                emb_flat = torch.zeros([input_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
                for i in range(len(self.cutoffs)):
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                    mask_i = (input_flat >= l_idx) & (input_flat < r_idx)
                    indices_i = mask_i.nonzero().squeeze()

                    if indices_i.numel() == 0:
                        continue
                    inp_i = input_flat.index_select(0, indices_i) - l_idx
                    emb_i = self.emb_layers[i](inp_i)
                    emb_i = F.linear(emb_i, self.emb_projs[i])

                    emb_flat.index_copy_(0, indices_i, emb_i)

                embed = emb_flat.view(*input.size(), self.d_proj)
            embed.mul_(self.emb_scale)
            return embed


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner, dropout, dropatt, tie_weight=True,
                 d_embed=None, div_val=1, tie_projs=[False], pre_lnorm=False, tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False, same_length=False, attn_type=0, clamp_len=-1, sample_softmax=-1):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()

        if attn_type == 0:  # default attn
            for i in range(n_layer):
                self.layers.append(RelPartialLearnabledecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, pre_lnorm=pre_lnorm))

        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(RelLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, pre_lnorm=pre_lnorm
                ))
        elif attn_type in [2, 3]:  # abs embedding
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(n_head, d_model, d_head, d_inner, dropout,
                                 dropatt=dropatt, pre_lnorm=pre_lnorm
                                 ))

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            pass


if __name__ == '__main__':
    pe = PositionalEmbedding(256)
    res = pe(3)
