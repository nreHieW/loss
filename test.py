import argparse
import torch
import torch.nn as nn
import torchvision
import numpy as np
import h5py
import os


def setup_surface_file(args, surf_file):
    f = h5py.File(surf_file, "w")
    xcoordinates = np.linspace(args.xmin, args.xmax, num=args.xnum)
    ycoordinates = np.linspace(args.ymin, args.ymax, num=args.ynum)
    f["xcoordinates"] = xcoordinates
    f["ycoordinates"] = ycoordinates
    f.close()


def eval_loss(net, criterion, dataloader, use_cuda):
    net.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss_sum += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if total > 2000:
                break

    return loss_sum / total, 100.0 * correct / total


def normalize_direction(direction, weights):
    for d, w in zip(direction, weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))


def normalize_directions_for_weights(direction, weights, norm="filter", ignore="biasbn"):
    assert len(direction) == len(weights)
    for name in direction.keys():
        d = direction[name]
        w = weights[name]
        if d.dim() <= 1:
            if ignore == "biasbn":
                d.fill_(0)
            else:
                d.copy_(w)
        else:
            normalize_direction(d, w)


def crunch(surf_file, net, w, dx, dy, dataloader, loss_key, acc_key, args):
    f = h5py.File(surf_file, "r+")
    xcoordinates = f["xcoordinates"][:]
    ycoordinates = f["ycoordinates"][:]
    losses = np.zeros((xcoordinates.shape[0], ycoordinates.shape[0]))
    accuracies = np.zeros((xcoordinates.shape[0], ycoordinates.shape[0]))

    criterion = nn.CrossEntropyLoss()

    # Normalize the directions
    normalize_directions_for_weights(dx, w, norm="filter")
    normalize_directions_for_weights(dy, w, norm="filter")

    for i, x in enumerate(xcoordinates):
        for j, y in enumerate(ycoordinates):
            new_weights = {}
            for name, param in net.state_dict().items():
                if name in w:
                    new_weights[name] = w[name] + x * dx[name] + y * dy[name]
                else:
                    new_weights[name] = param.clone()

            net.load_state_dict(new_weights)
            loss, acc = eval_loss(net, criterion, dataloader, args.cuda)
            losses[i, j] = loss
            accuracies[i, j] = acc
            print(f"Evaluating x={i+1}/{len(xcoordinates)}, y={j+1}/{len(ycoordinates)} coord=({x:.2f},{y:.2f}) loss={loss:.3f} acc={acc:.2f}")

    f[loss_key] = losses
    f[acc_key] = accuracies
    f.close()


def main():
    parser = argparse.ArgumentParser(description="Simplified loss surface plotting")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--model", default="resnet18", help="model name")
    parser.add_argument("--x", default="-1:1:10", help="A string with format xmin:xmax:xnum")
    parser.add_argument("--y", default="-1:1:10", help="A string with format ymin:ymax:ynum")
    parser.add_argument("--dir_type", default="weights", help="direction type: weights | states")
    parser.add_argument("--dataset", default="cifar10", help="dataset name")
    parser.add_argument("--datapath", default="./data", help="path to the dataset")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    args = parser.parse_args()

    args.xmin, args.xmax, args.xnum = map(float, args.x.split(":"))
    args.xnum = int(args.xnum)
    args.ymin, args.ymax, args.ynum = map(float, args.y.split(":"))
    args.ynum = int(args.ynum)

    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=args.datapath, train=True, download=True, transform=torchvision.transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    net = torchvision.models.__dict__[args.model](num_classes=10)
    if args.cuda:
        net.cuda()

    w = {name: param.data for name, param in net.named_parameters()}
    dx = {name: torch.randn_like(param) for name, param in net.named_parameters()}
    dy = {name: torch.randn_like(param) for name, param in net.named_parameters()}

    surf_file = f"loss_surface_{args.model}_{args.dataset}.h5"
    setup_surface_file(args, surf_file)
    crunch(surf_file, net, w, dx, dy, trainloader, "train_loss", "train_acc", args)


if __name__ == "__main__":
    main()
