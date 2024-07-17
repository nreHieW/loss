"""
References:
- https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
- https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py
- https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

Used timm version because it is easier to verify the implementation but I kept the einops where possible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


class FeedForward(nn.Module):
    """
    Implements the Feed Forward Layer of the Transformer

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

    def forward(self, x):
        out = self.ff1(x)
        out = self.act(out)
        out = self.drop(out)
        out = self.ff2(out)
        out = self.drop(out)
        return out


class Attention(nn.Module):
    """
    Implements the Multi-Head Attention Layer of the Transformer

    Args:
        dim: Dimension of the input and output
        heads: Number of heads
    """

    def __init__(self, dim, heads=8, attn_drop=0.0, proj_drop=0.0, bias: bool = True):
        super().__init__()
        self.head_dim = dim // heads
        self.heads = heads
        self.scale = self.head_dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    """
    Implements the Transformer Block

    Args:
        dim: Dimension of the input and output
        mlp_dim: Dimension of the hidden layer
        heads: Number of heads
        attn_drop: Dropout for the attention layer
        proj_drop: Dropout for the projection layer
        mlp_drop: Dropout for the hidden layer
        bias: Whether to use bias in the attention and projection layers
    """

    def __init__(
        self,
        dim,
        mlp_dim,
        heads: int = 8,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads=heads, attn_drop=attn_drop, proj_drop=proj_drop, bias=bias)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = FeedForward(dim=dim, hidden_dim=mlp_dim, drop=mlp_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class ViT(nn.Module):
    """
    Implements the Vision Transformer

    Args:
        height: Height of the input image
        width: Width of the input image
        patch_size: Size of the patch
        dim: Dimension of the input and output
        n_layers: Number of layers
        n_classes: Number of classes
        n_heads: Number of heads
        mlp_factor: Factor by which to multiply the dimension of the hidden layer
        channels: Number of channels in the input image
        attn_drop: Dropout for the attention layer
        proj_drop: Dropout for the projection layer
        mlp_drop: Dropout for the hidden layer
        bias: Whether to use bias in the attention and projection layers
    """

    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        dim: int,
        n_layers: int,
        n_classes: int,
        n_heads: int,
        mlp_factor: int,
        channels: int = 3,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert height % patch_size == 0 and width % patch_size == 0, "Invalid Patch Factor"

        self.patch_size = patch_size
        self.to_patch_embedding = PatchEmbedding(height, width, self.patch_size, dim, channels)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    mlp_dim=dim * mlp_factor,
                    heads=n_heads,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    mlp_drop=mlp_drop,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim), requires_grad=True)
        self.head = nn.Linear(dim, n_classes, bias=bias)
        self.n_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, dim))

    def forward(self, x):
        b_size = x.shape[0]
        x = self.to_patch_embedding(x)
        cls_token = self.cls_token.expand(b_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_end = x[:, 0]
        pred = self.head(cls_token_end)

        return pred


def _check_impl():
    # pip install timm
    import timm

    ref_model = timm.create_model("vit_base_patch16_384", pretrained=True)
    ref_model.eval()

    model = ViT(
        height=384,
        width=384,
        patch_factor=16,
        dim=768,
        n_layers=12,
        n_classes=1000,
        n_heads=12,
        mlp_factor=4,
        channels=3,
    )
    model.eval()

    x = torch.randn(1, 3, 384, 384)
    ref_y = ref_model(x)

    ref_model_n_params = sum(p.numel() for p in ref_model.parameters() if p.requires_grad)
    model_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert ref_model_n_params == model_n_params, f"Ref model has {ref_model_n_params} params, model has {model_n_params} params"

    ref_params = ref_model.named_parameters()
    model_params = model.named_parameters()

    dict_params_model = dict(model_params)
    model_params = model.named_parameters()

    for (model_n, model_p), (ref_n, ref_p) in zip(model_params, ref_params):
        assert model_p.shape == ref_p.shape, f"Model param {model_n} ({model_p.shape}) does not match ref param {ref_n} ({ref_p.shape})"
        dict_params_model[model_n].data.copy_(ref_p.data)

    y = model(x)
    assert torch.allclose(ref_y, y, atol=1e-4), f"Model output does not match ref output. Max diff: {torch.max(torch.abs(ref_y - y))}"

    return 0


if __name__ == "__main__":
    _check_impl()
