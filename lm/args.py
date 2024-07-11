from dataclasses import dataclass
import argparse


@dataclass
class TrainingArgs:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    num_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float

    model_type: str


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=128, help="block size")
    parser.add_argument("--n_layer", type=int, default=12, help="number of layers")
    parser.add_argument("--n_head", type=int, default=12, help="number of heads")
    parser.add_argument("--n_embd", type=int, default=768, help="embedding dimension")
    parser.add_argument("--num_steps", type=int, default=1000, help="number of steps")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight decay")
    parser.add_argument("--model_type", default="gpt", help="model type")
    args = parser.parse_args()
    return TrainingArgs(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_type=args.model_type,
    )
