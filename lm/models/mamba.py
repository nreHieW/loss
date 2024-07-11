from mamba_ssm import Mamba2
from gpt import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mamba = Mamba2(
            d_model=config.n_embd,
            d_state=64,
            d_conv=4,
            expand=2,
        )
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mamba(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MambaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tok_emb.weight = self.lm_head.weight
        self.config = config
        self.apply(self._init_weights)

    def _init_weights(self, module):
        print(module.state_dict().keys())
        # if isinstance(module, nn.Linear):
        #     std = 0.02
        #     if hasattr(module, "SCALE_INIT"):
        #         std *= (2 * self.config.n_layer) ** -0.5
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        #     if module.bias is not None:
        #         torch.nn.init.zeros_(module.bias)
        # elif isinstance(module, nn.Embedding):
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        pass

    def forward(self, idx, targets=None):
        B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        tok_emb = self.tok_emb(idx)
        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@dataclass
class MambaConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_embd: int


def get_mambalm(block_size, vocab_size, n_layer, n_embd):
    return MambaLM(MambaConfig(block_size=block_size, vocab_size=vocab_size, n_layer=n_layer, n_embd=n_embd))
