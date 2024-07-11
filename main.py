import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import copy
import numpy as np
import h5py
from torch.utils.data import DataLoader
from typing import Tuple, Dict


from reference.ref_resnet import load_ref

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
SEED = 42
torch.manual_seed(SEED)


def get_2_directions(model: nn.Module, verbose: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    params = model.named_parameters()
    dx = {}
    dy = {}
    for i, (name, param) in enumerate(params):
        curr_x = torch.randn_like(param)
        curr_y = torch.randn_like(param)
        if param.dim() <= 1:
            curr_x.fill_(0)
            curr_y.fill_(0)
        else:
            curr_x.mul_(param.norm() / (curr_x.norm() + 1e-10))
            curr_y.mul_(param.norm() / (curr_y.norm() + 1e-10))
        dx[name] = curr_x
        dy[name] = curr_y
    if verbose:
        _x = torch.cat([dx[name].flatten() for name in dx]).unsqueeze(0)
        _y = torch.cat([dy[name].flatten() for name in dy]).unsqueeze(0)
        similarity = F.cosine_similarity(_x, _y)
        print("cosine similarity between x-axis and y-axis: %f" % similarity)
    return dx, dy


def set_weights(model: nn.Module, original_state_dict: Dict[str, torch.Tensor], dx: Dict[str, torch.Tensor], dy: Dict[str, torch.Tensor], x_step: float, y_step: float) -> nn.Module:

    for name, param in model.named_parameters():
        change = x_step * dx[name] + y_step * dy[name]
        param.data = original_state_dict[name].to(DEVICE) + change.to(DEVICE)

    return model


def eval_loss(model: nn.Module, criterion: nn.Module, dataloader: DataLoader) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if total > 2000:
                break

    return loss_sum / total, 100.0 * correct / total


def crunch(
    model: nn.Module,
    original_state_dict: Dict[str, torch.Tensor],
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    dx: Dict[str, torch.Tensor],
    dy: Dict[str, torch.Tensor],
    criterion: nn.Module,
    trainloader: DataLoader,
    verbose: bool = True,
):
    res = {}
    for i, x in enumerate(x_coordinates):
        for j, y in enumerate(y_coordinates):
            new_model = set_weights(model, original_state_dict, dx, dy, x, y)
            loss, acc = eval_loss(new_model, criterion, trainloader)
            res[(i, j)] = (loss, acc)
            if verbose:
                print(f"Evaluating x={i+1}/{len(x_coordinates)}, y={j+1}/{len(y_coordinates)} coord=({x:.2f},{y:.2f}) loss={loss:.3f} acc={acc:.2f}")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--x", default="-1:1:10", help="A string with format xmin:x_max:xnum")
    parser.add_argument("--y", default="-1:1:10", help="A string with format ymin:ymax:ynum")
    parser.add_argument("--output_fpath", default="output.h5", help="output file path")
    args = parser.parse_args()
    x_min, x_max, x_num = map(int, args.x.split(":"))
    y_min, y_max, y_num = map(int, args.y.split(":"))

    model = load_ref().to(DEVICE)
    original_state_dict = copy.deepcopy(model.state_dict())

    dx, dy = get_2_directions(model)

    # download the dataset
    trainset = torchvision.datasets.CIFAR10(
        root="data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])]
        ),
    )
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False)

    ############## CRUNCH FUNCTION ################
    x_coordinates = torch.linspace(x_min, x_max, x_num)
    y_coordinates = torch.linspace(y_min, y_max, y_num)

    criterion = nn.CrossEntropyLoss()
    res = crunch(model, original_state_dict, x_coordinates, y_coordinates, dx, dy, criterion, trainloader)

    # save the results to the surface file
    with h5py.File(args.output_fpath, "w") as f:
        for (i, j), (loss, acc) in res.items():
            f[f"loss_{i}_{j}"] = loss
            f[f"acc_{i}_{j}"] = acc
        f.close()

    print("Results saved to %s" % args.output_fpath)
