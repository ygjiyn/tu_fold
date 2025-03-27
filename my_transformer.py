import torch
from torch import nn
import torch.nn.functional as F

import math


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 # === TransformerEncoderLayer param ===
                 d_model, n_head, 
                 # === TransformerEncoder param ===
                 n_layer, 
                 # === TransformerEncoderLayer param (with default value) ===
                 d_ff=2048, 
                 activation=F.relu, layer_norm_eps=1e-5, norm_first=False,
                 sdpa_dropout_p=0.1, sa_out_dropout_p=0.1, 
                 ff_hidden_dropout_p=0.1, ff_out_dropout_p=0.1, 
                 add_bias_layer_norm=True, add_bias_mha_qkv=True, 
                 add_bias_mha_out_proj=True, add_bias_ff=True,
                 # === TransformerEncoder param (with default value) ===
                 encoder_out_norm=None):
        super().__init__()
        self.encoder_out_norm = encoder_out_norm
        # instead of using deepcopy, we create Layers using a loop
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=norm_first,
                sdpa_dropout_p=sdpa_dropout_p,
                sa_out_dropout_p=sa_out_dropout_p,
                ff_hidden_dropout_p=ff_hidden_dropout_p,
                ff_out_dropout_p=ff_out_dropout_p,
                add_bias_layer_norm=add_bias_layer_norm,
                add_bias_mha_qkv=add_bias_mha_qkv,
                add_bias_mha_out_proj=add_bias_mha_out_proj,
                add_bias_ff=add_bias_ff
            ) 
            for _ in range(n_layer)
        ])

    def forward(self, x, mask=None, is_causal=False):
        # out = src
        for mod in self.layers:
            # TODO if attn_weight is needed, collect them from here
            # DONE think if I should return the the detached copy of attn_weight
            # I called .detach().cpu().numpy(), 
            # now the attn_weight is a numpy ndarray
            x, attn_weight = mod(x, src_mask=mask, is_causal=is_causal)
        if self.encoder_out_norm is not None:
            x = self.encoder_out_norm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff=2048, 
                 activation=F.relu, layer_norm_eps=1e-5, norm_first=False,
                 sdpa_dropout_p=0.1, sa_out_dropout_p=0.1, 
                 ff_hidden_dropout_p=0.1, ff_out_dropout_p=0.1, 
                 add_bias_layer_norm=True, add_bias_mha_qkv=True, 
                 add_bias_mha_out_proj=True, add_bias_ff=True):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model, 
            n_head, 
            sdpa_dropout_p=sdpa_dropout_p, 
            add_bias_qkv=add_bias_mha_qkv, 
            add_bias_out_proj=add_bias_mha_out_proj
        )
        self.linear_1 = nn.Linear(d_model, d_ff, bias=add_bias_ff)
        self.dropout_ff_hidden = nn.Dropout(ff_hidden_dropout_p)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=add_bias_ff)

        self.norm_first = norm_first
        # Using bias in LayerNorm requires pytorch >= 2.1
        self.norm_1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=add_bias_layer_norm)
        self.norm_2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=add_bias_layer_norm)
        self.dropout_sa_out = nn.Dropout(sa_out_dropout_p)
        self.dropout_ff_out = nn.Dropout(ff_out_dropout_p)

        self.activation = activation

    def forward(self, x, src_mask=None, is_causal=False):
        # in our implementation, we do not use src_key_padding_mask
        # since currently the implementation of sdpa in pytorch also does not use this
        # src_mask = F._canonical_mask(
        #     mask=src_mask,
        #     mask_name='src_mask',
        #     other_type=None,
        #     other_name='',
        #     target_type=src.dtype,
        #     check_other=False,
        # )
        # x = src
        if self.norm_first:
            sa_out, attn_weight = self._sa_block(
                self.norm_1(x), src_mask, is_causal=is_causal)
            x = x + sa_out
            x = x + self._ff_block(self.norm_2(x))
        else:
            sa_out, attn_weight = self._sa_block(
                x, src_mask, is_causal=is_causal)
            x = self.norm_1(x + sa_out)
            x = self.norm_2(x + self._ff_block(x))
        return x, attn_weight

    def _sa_block(self, x, attn_mask, is_causal=False):
        x, attn_weight = self.self_attn(
            x, 
            x, 
            x,
            attn_mask=attn_mask, 
            is_causal=is_causal
        )
        return self.dropout_sa_out(x), attn_weight
    
    def _ff_block(self, x):
        x = self.linear_2(self.dropout_ff_hidden(self.activation(self.linear_1(x))))
        return self.dropout_ff_out(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k=None, d_v=None, 
                 sdpa_dropout_p=0.0, 
                 add_bias_qkv=True, add_bias_out_proj=True):
        super().__init__()
        # attn_dropout_p is used in scaled_dot_product_attention
        # i.e., the dropout of attn_weight
        if d_k is None or d_v is None:
            # if either d_k or d_v is None
            # use the default setting in the transformer, which is
            # d_k or d_v = d_model / n_head
            assert d_model % n_head == 0, 'd_model must be divisible by n_head'
            # However, this is just a choice to make the computation cost
            # be similar to the single-head case (when both d_k and d_v is None), 
            # and is not a requirement
            # Therefore, it is not meaningful to check this 
            # when both d_k and d_v is not None
        self.n_head = n_head
        # use self.d_k instead of d_k in the following
        self.d_k = d_model // n_head if d_k is None else d_k
        # use self.d_v instead of d_v in the following
        self.d_v = d_model // n_head if d_v is None else d_v
        self.sdpa_dropout_p = sdpa_dropout_p

        # compute the projections of Q, K, V of multiple heads in parallel
        self.q_proj = nn.Linear(d_model, n_head * self.d_k, bias=add_bias_qkv)
        self.k_proj = nn.Linear(d_model, n_head * self.d_k, bias=add_bias_qkv)
        self.v_proj = nn.Linear(d_model, n_head * self.d_v, bias=add_bias_qkv)

        self.out_proj = nn.Linear(n_head * self.d_v, d_model, bias=add_bias_out_proj)

    def forward(self, q_mat, k_mat, v_mat, attn_mask=None, is_causal=False):
        # assume the input is batch_first, 
        # i.e., (batch, L, d_model)
        # after proj, (batch, L, n_head * d_k) -> (batch, L, n_head, d_k)
        # when compute sdpa, the matrix dimensions of query should be L * d_k
        # (batch, n_head, L, d_k)
        batch_size = q_mat.size(0)
        query = self.q_proj(q_mat).reshape(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        key = self.k_proj(k_mat).reshape(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.v_proj(v_mat).reshape(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)
        # after sdpa, attn_res is (batch, n_head, L, d_v)
        # is_causal=False, scale=None
        # DONE check self.training
        attn_res, attn_weight = scaled_dot_product_attention(
            query, key, value, is_train=self.training, 
            attn_mask=attn_mask, sdpa_dropout_p=self.sdpa_dropout_p, is_causal=is_causal)
        o_mat = attn_res.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        # out (batch, L, d_model)
        out = self.out_proj(o_mat)
        return out, attn_weight


def scaled_dot_product_attention(query, key, value, is_train, attn_mask=None, 
                                 sdpa_dropout_p=0.0, is_causal=False, scale=None):
    # matrix dimensions: query: L * d_k, key: S * d_k, value: S * d_v
    # batch dimensions: batch and head
    # L: length of query, S: length of key
    # the length of key and value should be the same (S)
    # the dim of the token (in a head) of query and key should be the same (d_k)
    # to conduct matmul
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    # add device here
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        # add device here
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float('-inf'))
        # "to" does not change inplace, change to an assignment
        # attn_bias.to(query.dtype)
        # it seems that masked_fill_ will not change the dtype of the original tensor
        # but transfer the fill value to the dtype of the original tensor
        # attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float('-inf'))
        else:
            attn_bias += attn_mask
    # matmul is only conducted on the matrix dimensions (last two)
    # after potential broadcasting on the batch dimensions
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    # make the sum of each "row" 1 (S entries), since value is S * d_v
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_weight = torch.dropout(attn_weight, sdpa_dropout_p, train=is_train)
    
    # also return the attn_weight
    # since the returned attn_weight is not included in any calculations
    # it will not influence the computing graph
    return attn_weight @ value, attn_weight.detach().cpu().numpy()

