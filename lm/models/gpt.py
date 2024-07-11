import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.up_proj = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.down_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.down_proj.SCALE_INIT = 1

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        fused = gate * self.up_proj(x)
        return self.down_proj(fused)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        tok_emb = self.tok_emb(idx)
        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


def get_gpt(block_size, vocab_size, n_layer, n_head, n_embd):
    config = GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
    return GPT(config)
