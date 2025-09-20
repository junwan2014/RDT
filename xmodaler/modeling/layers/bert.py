"""
Paper:  'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'
    - https://arxiv.org/pdf/1810.04805.pdf
	
From original at https://github.com/huggingface/transformers
Original copyright of Hugging Face team code below, modifications by Yehao Li, Copyright 2021.	
"""

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# tests directory-specific settings - this file is run automatically
# by pytest before any tests are run

import math

import numpy as np
import torch
from torch import nn
from torch import Tensor
from xmodaler.config import configurable
from ..layers.create_act import get_activation
from typing import Tuple
from einops import rearrange

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim=49, topk=16, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = 0.102
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
        self.routing_act_1 = nn.Softmax(dim=-2)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        # if not self.diff_routing:
        #     query, key = query.detach(), key.detach()
        # query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        # attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        # k = int(query_hat.size(2) * 0.76)
        # topk_attn_logit, topk_index = torch.topk(attn_logit, k, dim=-1)  # (n, p^2, k), (n, p^2, k)
        # r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)
        #
        # k = int(query_hat.size(2) * 0.9)
        # topk_attn_logit_1, topk_index_1 = torch.topk(attn_logit, k, dim=-2)  # (n, p^2, k), (n, p^2, k)
        # r_weight_1 = self.routing_act_1(topk_attn_logit_1)  # (n, p^2, k)

        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        k = int(key_hat.size(2) * 0.76)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        k = int(key_hat.size(2) * 1)
        low_k = int(key_hat.size(2)*0.5)
        topk_attn_logit_1, topk_index_1 = torch.topk(attn_logit, k, dim=-1)  # (n, p^2, k), (n, p^2, k)
        topk_index_1 = topk_index_1[:,:,:, low_k:-1]
        r_weight_1 = self.routing_act_1(topk_attn_logit_1)  # (n, p^2, k)


        return r_weight, topk_index, r_weight_1, topk_index_1


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()#160x8x49x64
        topk = r_idx.size(-1) #160x8x49x20
        # print(r_idx.size(), r_weight.size()) kv:160x8x1x49x64->160x8x49x49x64
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, p2, 1, w2, c_kv).expand(-1, -1, w2, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=3,
                               index=r_idx.view(n, p2, w2, topk, 1).expand(-1, -1, -1, -1, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class BertSelfAttention(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        qk_dim,
        scale,
        topk,
        diff_routing,
        param_routing,
        mul_weight
    ):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.qk_dim = qk_dim
        self.scale = scale
        self.topk = topk
        self.diff_routing = diff_routing
        self.param_routing = param_routing
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.query_d = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # self.transform = nn.Linear(2 * hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        self.dropout_1 = nn.Dropout(attention_probs_dropout_prob)
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        self.kv_gather = KVGather(mul_weight=mul_weight)
        self.attn_act = nn.Softmax(dim=-1)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_attention_heads": cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
            "attention_probs_dropout_prob": cfg.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB,
            "qk_dim": cfg.MODEL.BERT.QK_DIM,
            "scale": cfg.MODEL.BERT.SCALE,
            "topk": cfg.MODEL.BERT.TOPK,
            "diff_routing": cfg.MODEL.BERT.DIFF_ROUTING,
            "param_routing": cfg.MODEL.BERT.PARAM_ROUTING,
            "mul_weight": cfg.MODEL.BERT.MUL_WEIGHT
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)

        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)
        #return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, mode=None, history_states=None):
        mixed_query_layer = self.query(hidden_states)

        if history_states is not None:
            mixed_key_layer = self.key(history_states)
            mixed_value_layer = self.value(history_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) #160x8x49x64
        key_layer = self.transpose_for_scores(mixed_key_layer)  #160x8x49x64
        value_layer = self.transpose_for_scores(mixed_value_layer) #160x8x49x64
        #160x8x49x20 #160x8x49x64
        if mode=='v':
            r_weight, r_idx, _, r_idx_1 = self.router(query_layer, key_layer)

            # 正注意力
            r_idx_tmp = r_idx[:, :, 0, :].unsqueeze(-1).expand(r_weight.size(0), r_weight.size(1), r_weight.size(3),
                                                               query_layer.size(3))
            k_pix_sel = torch.gather(key_layer, 2, r_idx_tmp)  # 160x8x39x64
            v_pix_sel = torch.gather(value_layer, 2, r_idx_tmp)  # 160x8x39x64

            # 160x8x39x64 160x8x64x39 attention_scores: 160x8x39x20
            attention_scores = torch.matmul(query_layer, k_pix_sel.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                r_idx_tmp = r_idx[:, 0, 0, :]
                r_idx_tmp = r_idx_tmp.unsqueeze(-2).unsqueeze(-2)
                attention_mask_tmp = torch.gather(attention_mask, 3, r_idx_tmp)
                attention_scores = attention_scores + attention_mask_tmp

            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)
            context_layer = torch.matmul(attention_probs, v_pix_sel)  # 160x8x49x64

            shape_list = list(range(len(context_layer.shape)))
            shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
            context_layer = context_layer.permute(shape_list).contiguous()

            # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            r_idx_tmp_1 = r_idx_1[:, :, 0, :].unsqueeze(-1).expand(r_idx_1.size(0), r_idx_1.size(1), r_idx_1.size(3),
                                                                   query_layer.size(3))
            k_pix_sel_low = torch.gather(key_layer, 2, r_idx_tmp_1)  # 160x8x39x64
            v_pix_sel_low = torch.gather(value_layer, 2, r_idx_tmp_1)  # 160x8x39x64

            # 160x8x39x64 160x8x64x39 attention_scores: 160x8x39x20
            attention_scores_low = torch.matmul(query_layer, k_pix_sel_low.transpose(-2, -1))
            attention_scores_low = attention_scores_low / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                r_idx_tmp_1 = r_idx_1[:, 0, 0, :]
                r_idx_tmp_1 = r_idx_tmp_1.unsqueeze(-2).unsqueeze(-2)
                attention_mask_low = torch.gather(attention_mask, 3, r_idx_tmp_1)
                attention_scores_low = attention_scores_low + attention_mask_low

            attention_probs_low = nn.Softmax(dim=-1)(attention_scores_low)
            attention_probs_low = self.dropout_1(attention_probs_low)
            context_layer_1 = torch.matmul(attention_probs_low, v_pix_sel_low)  # 160x8x49x64

            context_layer_1 = context_layer_1.permute(shape_list).contiguous()
            context_layer_1 = context_layer_1.view(*new_context_layer_shape)
            # fusion
            # context_layer = torch.cat((context_layer, context_layer_1), dim=-1)
            # context_layer = self.transform(context_layer)
            context_layer = context_layer + 0.2*context_layer_1




        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer)
            shape_list = list(range(len(context_layer.shape)))
            shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
            context_layer = context_layer.permute(shape_list).contiguous()  # 160x49x8x64
            # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs

class BertSelfOutput(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size: int,
        layer_norm_eps: float,
        hidden_dropout_prob: float
    ):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": cfg.MODEL.BERT.HIDDEN_DROPOUT_PROB
        }

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        bert_self_attention,
        bert_self_output
    ):
        super(BertAttention, self).__init__()
        self.self = bert_self_attention
        self.output = bert_self_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_self_attention": BertSelfAttention(cfg),
            "bert_self_output": BertSelfOutput(cfg),
        }

    def forward(self, input_tensor, attention_mask, mode=None, history_states=None):
        self_output, attention_probs = self.self(input_tensor, attention_mask, mode, history_states)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs

class BertIntermediate(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size: int,
        hidden_act: str,
        intermediate_size: int,
        intermediate_drop: float
    ):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = get_activation(hidden_act)
        self.dropout = nn.Dropout(intermediate_drop)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "hidden_act": cfg.MODEL.BERT.HIDDEN_ACT,
            "intermediate_size": cfg.MODEL.BERT.INTERMEDIATE_SIZE,
            "intermediate_drop": cfg.MODEL.BERT.INTERMEDIATE_DROP
        }

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size: int,
        intermediate_size: int,
        layer_norm_eps: float,
        ffn_dropout_prob: float
    ):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(ffn_dropout_prob)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "intermediate_size": cfg.MODEL.BERT.INTERMEDIATE_SIZE,
            "layer_norm_eps": 1e-12,
            "ffn_dropout_prob": cfg.MODEL.BERT.FFN_DROPOUT_PROB
        }

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertXAttention(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob
    ):
        super(BertXAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "num_attention_heads": cfg.MODEL.BERT.NUM_ATTENTION_HEADS,
            "attention_probs_dropout_prob": cfg.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB
        }

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        #return x.permute(0, 2, 1, 3)
        shape_list = list(range(len(new_x_shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        return x.permute(shape_list)

    def forward(self, query, key, value, attention_mask):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        shape_list = list(range(len(context_layer.shape)))
        shape_list[-2], shape_list[-3] = shape_list[-3], shape_list[-2]
        context_layer = context_layer.permute(shape_list).contiguous()

        #context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

class BertCrossAttention(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        bert_cross_attention,
        bert_self_output
    ):
        super(BertCrossAttention, self).__init__()
        self.self = bert_cross_attention
        self.output = bert_self_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_cross_attention": BertXAttention(cfg),
            "bert_self_output": BertSelfOutput(cfg),
        } 

    def forward(self, query, key, value, attention_mask, q_attention_mask):
        x_output, attention_probs = self.self(query, key, value, attention_mask)
        attention_output = self.output(x_output, query)
        return attention_output, attention_probs


class BertLayer(nn.Module):
    @configurable
    def __init__(
        self, 
        *,
        bert_attention, 
        bert_intermediate,
        bert_output
    ):
        super(BertLayer, self).__init__()
        self.attention = bert_attention
        self.intermediate = bert_intermediate
        self.output = bert_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_attention": BertAttention(cfg),
            "bert_intermediate": BertIntermediate(cfg),
            "bert_output": BertOutput(cfg)
        }

    def forward(self, hidden_states, attention_mask, mode=None, history_states=None):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask, mode, history_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

class BertUnderstandingLayer(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        bert_attention,
        v_bert_intermediate,
        v_bert_output,
        t_bert_intermediate,
        t_bert_output,
    ):
        super(BertUnderstandingLayer, self).__init__()
        self.biattention = bert_attention
        self.v_intermediate = v_bert_intermediate
        self.v_output = v_bert_output
        self.t_intermediate = t_bert_intermediate
        self.t_output = t_bert_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_attention": BertAttention(cfg),
            "v_bert_intermediate": BertIntermediate(cfg),
            "v_bert_output": BertOutput(cfg),
            "t_bert_intermediate": BertIntermediate(cfg),
            "t_bert_output": BertOutput(cfg)
        }

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2):
        att_len = attention_mask1.shape[-1]
        feats = torch.cat([input_tensor1, input_tensor2], dim=1)
        attention_mask = torch.cat([attention_mask1, attention_mask2], dim=-1)
        feats, _ = self.biattention(feats, attention_mask)

        v_attention_output = feats[:, :att_len]
        t_attention_output = feats[:, att_len:]

        v_intermediate_output = self.v_intermediate(v_attention_output)
        v_feats = self.v_output(v_intermediate_output, v_attention_output)

        t_intermediate_output = self.t_intermediate(t_attention_output)
        t_feats = self.t_output(t_intermediate_output, t_attention_output)

        return v_feats, t_feats

class BertGenerationLayer(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        bert_attention,
        bert_cross_attention,
        bert_intermediate,
        bert_output
    ):
        super(BertGenerationLayer, self).__init__()
        self.self_attn = bert_attention
        self.x_att = bert_cross_attention
        self.intermediate = bert_intermediate
        self.output = bert_output

    @classmethod
    def from_config(cls, cfg):
        return {
            "bert_attention": BertAttention(cfg),
            "bert_cross_attention": BertCrossAttention(cfg),
            "bert_intermediate": BertIntermediate(cfg),
            "bert_output": BertOutput(cfg)
        }

    def forward(self, lang_feats, v_feats, lang_attention_mask=None, v_attention_mask=None, mode=None, t_history_states=None):
        x, _ = self.self_attn(lang_feats, lang_attention_mask, mode, t_history_states)
        x, _ = self.x_att(x, v_feats, v_feats, v_attention_mask, lang_attention_mask)
        intermediate_output = self.intermediate(x)
        layer_output = self.output(intermediate_output, x)

        return layer_output

class BertPooler(nn.Module):
    @configurable
    def __init__(
        self, 
        *, 
        hidden_size: int
    ):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE
        }

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        hidden_size: int,
        hidden_act: str,
        layer_norm_eps: float
    ):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = get_activation(hidden_act)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    @classmethod
    def from_config(cls, cfg):
        return {
            "hidden_size": cfg.MODEL.BERT.HIDDEN_SIZE,
            "hidden_act": cfg.MODEL.BERT.HIDDEN_ACT,
            "layer_norm_eps": 1e-12,
        }

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



