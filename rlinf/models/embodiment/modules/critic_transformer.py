import torch
import torch.nn as nn


class SeqQFuncTorch(nn.Module):
    def __init__(
        self,
        state_dim,
        image_dim,
        action_dim,
        n_embed,
        n_heads,
        n_layer,
        action_horizon,
        dropout_rate,
        num_q_heads=1,
    ):
        super().__init__()
        self.context_proj = nn.Linear(state_dim + image_dim, n_embed, bias=False)
        self.action_embed = nn.Linear(action_dim, n_embed, bias=False)
        self.pos_embed = nn.Embedding(1 + action_horizon, n_embed)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embed,
            nhead=n_heads,
            dim_feedforward=4 * n_embed,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, num_q_heads)

    def forward(self, state_features, image_features, actions):
        """
        Args:
            state_features:  [B, state_dim]
            image_features:  [B, image_dim]
            actions:         [B, T, action_dim]
        Returns:
            q_values:        [B, T, num_q_heads]
        """
        B, T, _ = actions.shape

        context = torch.cat([state_features, image_features], dim=-1)  # [B, state_dim+image_dim]
        context_emb = self.context_proj(context).unsqueeze(1)          # [B, 1, n_embed]
        action_emb = self.action_embed(actions)                        # [B, T, n_embed]
        seq = torch.cat([context_emb, action_emb], dim=1)              # [B, 1+T, n_embed]

        pos = self.pos_embed(torch.arange(1 + T, device=seq.device))   # [1+T, n_embed]
        x = seq + pos

        x = self.transformer(x)
        x = self.norm(x)
        x = self.head(x)            # [B, 1+T, out_dim]
        return x[:, 1:, :]          # drop context token, keep action Q-values



"""
Adapted from Nano GPT implementation in https://github.com/karpathy/nanoGPT

Main adaptations include:
1. Adapt the input from language model to critic model of RL
2. Add several save and load functions for NN weights
3. Add support for changing device and dtype
"""

import math
import torch
import torch.nn as nn
from addict import Dict


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd,
                                bias=config.bias)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,
                                bias=config.bias)  # Note: needs special init

        # regularization
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape[:-2], x.shape[-2], x.shape[-1]

        q, k, v = self.c_attn(x).split(self.n_embd, dim=-1)

        # (*B, T, n_head, C / n_head) -> (*B, n_head, T, C / n_head)
        k = k.view(*B, T, self.n_head, C // self.n_head).transpose(-3, -2)
        q = q.view(*B, T, self.n_head, C // self.n_head).transpose(-3, -2)
        v = v.view(*B, T, self.n_head, C // self.n_head).transpose(-3, -2)

        dropout_p = self.dropout if self.training else 0

        # Causal self-attention, Shape of y (*B, nh, T, d_k)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p,
            is_causal=True)

        # (*B, n_head, T, d_k) -> (*B, T, n_head, d_k)
        # -> (*B, T, n_head, C)
        y = y.transpose(-3, -2).contiguous().view(*B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        bias = config.bias
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.use_layer_norm = config.use_layer_norm
        if self.use_layer_norm:
            self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.use_layer_norm:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.attn(x)
            x = x + self.mlp(x)
        return x

def parse_dtype_device(dtype: str, device: str):
    """
    Parse data type and device from string input

    Args:
        dtype: data type string
        device: device string

    Returns:
        dtype and device used to initialize a tensor
    """
    if dtype == "float32" or dtype == "torch.float32":
        target_dtype = torch.float32
    elif dtype == "float64" or dtype == "torch.float64":
        target_dtype = torch.float64
    else:
        raise NotImplementedError

    target_device = torch.device(device)
    return target_dtype, target_device


class CriticGPT(nn.Module):

    def __init__(self, **config):
        super().__init__()

        config = Dict(config)
        self.config = config
        self.use_layer_norm = config.use_layer_norm

        module_dict = dict(
            action_encoder=nn.Linear(config.action_dim, config.n_embd,
                                     bias=False),
            pos_enc=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        )

        if not self.use_layer_norm:
            del module_dict['ln_f']

        self.transformer = nn.ModuleDict(module_dict)
        self.output_layer = nn.Linear(config.n_embd, 1, bias=False)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        # device and dtype?
        dtype, device = parse_dtype_device(config.dtype, config.device)
        self.dtype, self.device = dtype, device

        self.transformer.to(device=device, dtype=dtype)
        self.output_layer.to(device=device, dtype=dtype)

        self.gpt_name = config.name + "_gpt"

        self.relative_pos = config.relative_pos

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, state_features, image_features, actions, idx_s, idx_a):
        """
        Compute the value of one state and a sequence of actions

        Args:
            state_features: current state, [*add_dim, state_dim]
            image_features: current image, [*add_dim, image_dim]
            actions: action sequences, [*add_dim, num_actions, action_dim]
            idx_s: time step index of current state, [*add_dim]
            idx_a: time step indices of actions, [*add_dim, num_actions]

            The idx_s and idx_a are used to compute the positional embeddings

        Returns:
            The output has multiple tokens, with the first token the V-func, and
            the rest the Q-funcs. The shape is [*add_dim, num_actions + 1]
        """

        if actions is None:
            assert idx_a is None
            t = 0
        else:
            t = actions.size(-2)

        assert t + 1 <= self.config.block_size

        state_emd = state_features.unsqueeze(-2) # [*add_dim, 1, hidden_dim]
        image_emd = image_features.unsqueeze(-2) # [*add_dim, 1, hidden_dim]
        if actions is not None:
            action_emd = self.transformer.action_encoder(actions)

            # Shape [*add_dim, num_actions + 2, n_embed]
            seq_emb = torch.cat([state_emd, image_emd, action_emd], dim=-2)

            # Shape [*add_dim, num_actions + 2]
            seq_pos = torch.cat([idx_s[..., None], idx_a], dim=-1)

        else:
            seq_emb = torch.cat([state_emd, image_emd], dim=-2)
            seq_pos = idx_s[..., None]

        # If relative positional embedding is used,
        # then state position is always 0, and action positions are 1, 2, ...
        if self.relative_pos:
            # shape (2, num_actions + 2)
            seq_pos = torch.arange(0, 2 + t, dtype=torch.long, # 2 for state and image
                                   device=self.device)[None]

        # Shape [*add_dim, num_actions + 2, n_embed]
        seq_pos_emb = self.transformer.pos_enc(seq_pos)

        x = self.transformer.drop(seq_emb + seq_pos_emb)

        for block in self.transformer.h:
            x = block(x)

        if self.use_layer_norm:
            x = self.transformer.ln_f(x)

        # Shape [*add_dim, num_actions + 2, n_embed]
        # -> Shape [*add_dim, num_actions + 2, 1] -> [*add_dim, num_actions + 2]
        x = self.output_layer(x).squeeze(-1)  # value is dimensionless

        # v = x[..., 0]  # shape [*add_dim]
        # q = x[..., 2:]  # shape [*add_dim, num_actions]
        return x
