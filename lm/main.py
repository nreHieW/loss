import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import numpy as np
import h5py
from tqdm import tqdm
from typing import Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device", DEVICE)

SEED = 42

torch.manual_seed(SEED)
from torch.utils.data import IterableDataset


class MiniPile(IterableDataset):
    def __init__(self, ds, tokenizer, total_tokens=1e6, batch_size=1):
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.eos_token_id = tokenizer.eos_token_id
        self.batch_size = batch_size

        self.tokens = []
        buffer = []
        total_length = 0
        for i in tqdm(range(len(ds))):
            item = ds[i]["text"]
            tokens = tokenizer(item)["input_ids"]
            tokens.append(self.eos_token_id)
            if total_length + len(tokens) > self.max_length:
                self.tokens.extend(buffer)
                buffer = tokens
                total_length = len(tokens)
            else:
                buffer.extend(tokens)
                total_length += len(tokens)
            if len(self.tokens) > total_tokens:
                break

        if buffer and (not total_tokens or len(self.tokens) < total_tokens):
            remaining = total_tokens - len(self.tokens) if total_tokens else len(buffer)
            self.tokens.extend(buffer[:remaining])

        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.tokens) - self.max_length:
            raise StopIteration

        batch = []
        for _ in range(self.batch_size):
            if self.current_index >= len(self.tokens) - self.max_length:
                break

            chunk = self.tokens[self.current_index : self.current_index + self.max_length + 1]
            input_sequence = torch.tensor(chunk[:-1])
            labels = torch.tensor(chunk[1:])

            batch.append(
                {
                    "input_ids": input_sequence,
                    "labels": input_sequence,
                }
            )

            self.current_index += self.max_length

        if not batch:
            raise StopIteration

        return self._collate_batch(batch)

    def _collate_batch(self, batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }

    def __len__(self):
        return (len(self.tokens) - self.max_length) // (self.max_length * self.batch_size)

    def reset(self):
        self.current_index = 0


def model_handler(model_type: str):
    if model_type == "gpt2":
        name = "openai-community/gpt2"
    elif model_type == "mamba":
        name = "state-spaces/mamba-130m-hf"

    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = 1024
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(DEVICE)
    # model = None

    return model, tokenizer


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


def eval_loss(model: nn.Module, dataset: MiniPile) -> float:
    model.eval()
    loss = 0.0
    i = 0
    dataset.reset()

    with torch.no_grad():
        for item in dataset:
            input_ids = item["input_ids"].to(DEVICE)
            labels = item["labels"].to(DEVICE)

            out = model(input_ids=input_ids, labels=labels)
            i += 1
            loss += out.loss.item()

    return loss / i


def crunch(
    model: nn.Module,
    original_state_dict: Dict[str, torch.Tensor],
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    dx: Dict[str, torch.Tensor],
    dy: Dict[str, torch.Tensor],
    dataset: MiniPile,
    verbose: bool = True,
):
    res = {}
    # starting
    loss = eval_loss(model, dataset=dataset)
    print(f"Initial loss={loss:.3f}")
    for i, x in enumerate(x_coordinates):
        for j, y in enumerate(y_coordinates):
            new_model = set_weights(model, original_state_dict, dx, dy, x, y)
            loss = eval_loss(new_model, dataset=dataset)
            res[(i, j)] = loss
            if verbose:
                print(f"Evaluating x={i+1}/{len(x_coordinates)}, y={j+1}/{len(y_coordinates)} coord=({x:.2f},{y:.2f}) loss={loss:.3f}")

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--x", default="-1:1:40", help="A string with format xmin:x_max:xnum")
    parser.add_argument("--y", default="-1:1:40", help="A string with format ymin:ymax:ynum")
    parser.add_argument("--model_type", default="gpt2", help="model type")
    parser.add_argument("--output_fpath", default="output.h5", help="output file path")
    args = parser.parse_args()
    x_min, x_max, x_num = map(int, args.x.split(":"))
    y_min, y_max, y_num = map(int, args.y.split(":"))

    model, tokenizer = model_handler(args.model_type)
    original_state_dict = copy.deepcopy(model.state_dict())

    dx, dy = get_2_directions(model)

    ds = load_dataset("JeanKaddour/minipile", split="test")

    dataset = MiniPile(ds=ds, tokenizer=tokenizer, batch_size=args.batch_size)
    print("Tokens", len(dataset.tokens))
    # dataset = ds

    ############## CRUNCH FUNCTION ################
    x_coordinates = torch.linspace(x_min, x_max, x_num)
    y_coordinates = torch.linspace(y_min, y_max, y_num)

    criterion = nn.CrossEntropyLoss()
    res = crunch(model, original_state_dict, x_coordinates, y_coordinates, dx, dy, dataset)

    output_fpath = f"{args.model_type}_{args.output_fpath}"
    # save the results to the surface file
    with h5py.File(output_fpath, "w") as f:
        for (i, j), loss in res.items():
            f[f"loss_{i}_{j}"] = loss
        f.close()

    print("Results saved to %s" % args.output_fpath)
