import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class ResidualBlock(nn.Module):
    """
    Implements a residual block as described in Figure 5 (left) of the paper

    Args:
        in_channels (int): number of input channels
        first_out_channels (int): number of output channels of the first convolution
        stride (int): stride of the first convolution
        identity_method (str): either "A" or "B" as described in section 3.3 of the paper.
        A refers to padding the residual with zeros and B refers to a 1x1 convolution
    """

    # Resnet 18 and 34 have no bottlneck (ie 2 layers per block)
    def __init__(
        self,
        in_channels: int,
        first_out_channels: int,
        stride: int,
        identity_method: str = "A",
    ):
        super().__init__()
        if in_channels == first_out_channels:  # second in a block
            stride = 1
        self.out_channels = first_out_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_out_channels,
            kernel_size=(3, 3),
            stride=stride,
            bias=False,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(first_out_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=first_out_channels,
            out_channels=first_out_channels,
            kernel_size=(3, 3),
            stride=1,
            bias=False,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(first_out_channels)

        self.is_downsample = False
        self.identity_method = identity_method
        if stride != 1:
            if identity_method == "B":
                # downsampling Method B in section 3.3 of the paper
                self.identity = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=first_out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(first_out_channels),
                )
            self.is_downsample = True

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.is_downsample:
            if self.identity_method == "A":
                residual = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, self.out_channels - x.shape[1]))
            elif self.identity_method == "B":
                residual = self.identity(x)
        else:
            residual = x

        out += residual
        out = self.act(out)
        return out


class BottleneckBlock(nn.Module):
    """
    Implements a bottleneck block as described in Figure 5 (right) of the paper

    Args:
        in_channels (int): number of input channels
        first_out_channels (int): number of output channels of the first convolution
        stride (int): stride of the first convolution
        identity_method (str): either "A" or "B" as described in section 3.3 of the paper.
        A refers to padding the residual with zeros and B refers to a 1x1 convolution
    """

    def __init__(
        self,
        in_channels: int,
        first_out_channels: int,
        stride: int,
        identity_method: str = "A",
    ):
        super().__init__()
        out_channels = first_out_channels * 4
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_out_channels,
            kernel_size=(1, 1),
            stride=1,
            bias=False,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(first_out_channels)

        # https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
        self.conv2 = nn.Conv2d(
            in_channels=first_out_channels,
            out_channels=first_out_channels,
            kernel_size=(3, 3),
            stride=stride,
            bias=False,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(first_out_channels)
        self.conv3 = nn.Conv2d(
            in_channels=first_out_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            bias=False,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        self.is_downsample = False
        self.identity_method = identity_method
        if in_channels != out_channels:
            if identity_method == "B":
                self.identity = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            self.is_downsample = True

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.is_downsample:
            if self.identity_method == "A":
                residual = F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, self.out_channels - x.shape[1]))
            elif self.identity_method == "B":
                residual = self.identity(x)
        else:
            residual = x

        out += residual
        out = self.act(out)
        return out


class ResNet(nn.Module):
    """
    Implements a ResNet as described in the paper

    Args:
        num_blocks (list): number of blocks per layer
        num_classes (int): number of classes
        block_fn (nn.Module): either ResidualBlock or BottleneckBlock
        dimensions (list): list of dimensions for each layer
        first_kernel_size (int): kernel size of the first convolution
        identity_method (str): either "A" or "B" as described in section 3.3 of the paper.
        A refers to padding the residual with zeros and B refers to a 1x1 convolution
        special_init (bool): whether to use kaiming initialization or not

    """

    def __init__(
        self,
        num_blocks: list,
        num_classes: int,
        block_fn: str,
        dimensions: list,
        first_kernel_size: int,
        identity_method: str,
        special_init: bool = True,
    ):
        super().__init__()
        # conv2d: output_size = (input_size - filter_size + 2*padding)/stride + 1
        padding = 3 if first_kernel_size == 7 else 1  # Imagenet or Cifar
        stride = 2 if first_kernel_size == 7 else 1
        self.block_fn = ResidualBlock if block_fn == "residual" else BottleneckBlock
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=dimensions[0],
            kernel_size=first_kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # half input size imagneet 224 -> 112
        self.bn1 = nn.BatchNorm2d(dimensions[0])
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        assert len(dimensions) == len(num_blocks), "There should be 4 layers of blocks"

        prev_dim = dimensions[0]

        layers = []
        for dim, n in zip(dimensions, num_blocks):
            for i in range(n):
                layers.append(
                    self.block_fn(
                        in_channels=prev_dim,
                        first_out_channels=dim,
                        stride=2 if ((i == 0) and (dim != dimensions[0])) else 1,
                        identity_method=identity_method,
                    )
                )
                prev_dim = layers[-1].out_channels

        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layers[-1].out_channels, num_classes)

        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L167
        if special_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, BottleneckBlock) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, ResidualBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.pool1(out)

        out = self.layers(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


IMAGENET_MODEL_PARAMS = {
    "18": {
        "block_fn": "residual",
        "num_blocks": [2, 2, 2, 2],
        "first_kernel_size": 7,
        "dimensions": [64, 128, 256, 512],
        "identity_method": "B",
    },
    "34": {
        "block_fn": "residual",
        "num_blocks": [3, 4, 6, 3],
        "first_kernel_size": 7,
        "dimensions": [64, 128, 256, 512],
        "identity_method": "B",
    },
    "50": {
        "block_fn": "bottleneck",
        "num_blocks": [3, 4, 6, 3],
        "first_kernel_size": 7,
        "dimensions": [64, 128, 256, 512],
        "identity_method": "B",
    },
    "101": {
        "block_fn": "bottleneck",
        "num_blocks": [3, 4, 23, 3],
        "first_kernel_size": 7,
        "dimensions": [64, 128, 256, 512],
        "identity_method": "B",
    },
    "152": {
        "block_fn": "bottleneck",
        "num_blocks": [3, 8, 36, 3],
        "first_kernel_size": 7,
        "dimensions": [64, 128, 256, 512],
        "identity_method": "B",
    },
}


def _check_impl():
    for model_name, config in IMAGENET_MODEL_PARAMS.items():
        ref_model = torch.hub.load("pytorch/vision:v0.10.0", f"resnet{model_name}", pretrained=True)
        ref_model.eval()
        x = torch.randn(1, 3, 224, 224)  # imagenet size
        out1 = ref_model(x)
        model = ResNet(
            num_classes=1000,
            **config,
        )
        model.eval()
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

        out2 = model(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    return 0


if __name__ == "__main__":
    _check_impl()
