import torch
import torch.nn as nn

from functools import partial
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba
from timm.models.layers import DropPath


class BasicBlock(nn.Module):
    """
    Note:
    In the original paper, the authors used Add -> LN -> Mixer (Mamba) for performance reasons because they fuse the
    Add and LN operations into a single operation. However, I didn't find any performance speedups. So for this implementation,
    I follow standard VIT implementation and use LN -> Mixer (Mamba) -> Add
    """

    def __init__(self, dim: int, ssm_drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.mamba = Mamba(
            d_model=dim,
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(ssm_drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        hidden_states = self.norm(x)
        hidden_states = self.mamba(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + x
        hidden_states = self.drop_path(hidden_states)
        return hidden_states


class PatchEmbedding(nn.Module):
    def __init__(self, height: int, width: int, patch_size: int, dim: int, in_channels):
        super().__init__()
        # takes in (b, c, h, w) -> (b, l, d)
        self.n_patches = (height / patch_size) * (width / patch_size)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        out = self.conv(x)  # (b, c, h, w) -> (b, dim, h', w') where h' * w' = L
        out = out.flatten(2)  # (b, dim, L)
        out = out.transpose(1, 2)  # (b, L, dim)
        return out


class BiDirectionalConcatBlock(nn.Module):
    def __init__(self, dim: int, ssm_drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.out_dim = dim
        self.dim = dim // 2
        self.mamba1 = Mamba(
            d_model=self.dim,
        )
        self.mamba2 = Mamba(
            d_model=self.dim,
        )
        self.norm = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(ssm_drop)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        hidden_states = self.norm(x)
        rev_input = torch.flip(hidden_states, dims=[1])
        forward_states = self.mamba1(hidden_states)
        backward_states = self.mamba2(rev_input)
        hidden_states = torch.cat([forward_states, backward_states], dim=-1)
        hidden_states = self.drop_path(hidden_states)
        hidden_states = hidden_states + x
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BiDirectionalAddBlock(nn.Module):
    def __init__(self, dim: int, ssm_drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.mamba1 = Mamba(
            d_model=self.dim,
        )
        self.mamba2 = Mamba(
            d_model=self.dim,
        )
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(ssm_drop)
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        hidden_states = self.norm(x)
        rev_input = torch.flip(hidden_states, dims=[1])
        forward_states = self.mamba1(hidden_states)
        backward_states = self.mamba2(rev_input)
        hidden_states = forward_states + backward_states
        hidden_states = self.drop_path(hidden_states)
        hidden_states = hidden_states + x
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """
    Implements the Feed Forward Layer

    Args:
        dim: Dimension of the input and output
        hidden_dim: Dimension of the hidden layer
    """

    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.ff1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        out = self.ff1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.ff2(out)
        out = self.drop(out)
        return out


class BiDirectionalAddFFBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ssm_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_factor: int = 4,
    ) -> None:
        super().__init__()
        self.block = BiDirectionalAddBlock(dim, ssm_drop, drop_path)
        self.ff = FeedForward(dim, dim * mlp_factor, mlp_drop)

    def forward(self, x):
        x = x + self.block(x)
        x = x + self.ff(x)
        return x


class BiDirectionalConcatFFBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ssm_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_factor: int = 4,
    ) -> None:
        super().__init__()
        self.block = BiDirectionalConcatBlock(dim, ssm_drop, drop_path)
        self.ff = FeedForward(dim, dim * mlp_factor, mlp_drop)

    def forward(self, x):
        x = x + self.block(x)
        x = x + self.ff(x)
        return x


class MambaBackbone(nn.Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        block_type: str,
        ssm_drop: float = 0.0,
        mlp_factor: int = 4,
        mlp_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if block_type == "basic":
            block_fn = BasicBlock
        elif block_type == "bi_concat":
            block_fn = BiDirectionalConcatBlock
        elif block_type == "bi_add":
            block_fn = BiDirectionalAddBlock
        elif block_type == "bi_concat_ff":
            block_fn = partial(BiDirectionalConcatFFBlock, mlp_drop=mlp_drop, mlp_factor=mlp_factor)
        elif block_type == "bi_add_ff":
            block_fn = partial(BiDirectionalAddFFBlock, mlp_drop=mlp_drop, mlp_factor=mlp_factor)
        else:
            raise NotImplementedError(f"Block {block_type} not implemented")
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    ssm_drop=ssm_drop,
                    drop_path=drop_path,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VisionMamba(nn.Module):
    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        dim: int,
        n_layers: int,
        n_classes: int,
        block_type: str,
        pos_emb: bool,
        channels: int = 3,
        ssm_drop: float = 0.0,
        mlp_drop: float = 0.0,
        mlp_factor: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert height % patch_size == 0 and width % patch_size == 0, "Invalid Patch Factor"

        self.patch_size = patch_size
        self.to_patch_embedding = PatchEmbedding(height, width, self.patch_size, dim, channels)

        self.backbone = MambaBackbone(n_layers, dim, block_type, ssm_drop, mlp_factor, mlp_drop, drop_path)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.head = nn.Linear(dim, n_classes)
        self.n_patches = (height // patch_size) * (width // patch_size)
        self.use_pos_emb = pos_emb
        if pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))

    def forward(self, x):
        b_size = x.shape[0]
        x = self.to_patch_embedding(x)
        cls_token = self.cls_token.expand(b_size, -1, -1)
        x = torch.cat((x, cls_token), dim=1)
        if self.use_pos_emb:
            x = x + self.pos_embed

        x = self.backbone(x)
        cls_token_end = x[:, -1]
        pred = self.head(cls_token_end)

        return pred
