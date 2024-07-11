import torch
import requests
from models.gpt import get_gpt
from models.mamba import get_mambalm
from args import get_args


def get_model(model_name: str, block_size, vocab_size, n_layer, n_head, n_embd):
    if model_name == "gpt":
        return get_gpt(block_size, vocab_size, n_layer, n_head, n_embd)
    elif model_name == "mamba":
        return get_mambalm(block_size, vocab_size, n_layer, n_embd)


class ShakespeareDataset:
    def __init__(self, block_size: int):
        self.url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        self.text = requests.get(self.url).text
        self.chars = list(set(self.text))
        self.data = self.text
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.self.block_size = block_size
        self.create_dataset()

    def create_dataset(self):
        data = torch.tensor(self.encode(self.data), dtype=torch.long)
        n = int(len(data) * 0.9)
        self.train_data, self.valid_data = data[:n], data[n:]

    def get_batch(self, split: str, batch_size: int):
        data = self.train_data if split == "train" else self.valid_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, text):
        return "".join([self.idx_to_char[i] for i in text])


def configure_optimizers(self, weight_decay, learning_rate):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [{"params": decay_params, "weight_decay": weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer


def evaluate(model, dataset, batch_size, num_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss = 0.0
    iters = 0
    with torch.no_grad():
        for _ in range(num_steps):
            x, y = dataset.get_batch("valid", batch_size)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x)
            total_loss += loss.item()
            iters += 1
    return total_loss / iters


def train(model, dataset, num_steps, batch_size, learning_rate, weight_decay, eval_interval=500, eval_steps=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    optim = configure_optimizers(model, weight_decay, learning_rate)

    for step in range(num_steps):
        x, y = dataset.get_batch("train", batch_size)
        x, y = x.to(device), y.to(device)

        logits, loss = model(x)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % eval_interval == 0:
            eval_loss = evaluate(model, dataset, batch_size, eval_steps)  # 200 batches for evaluation
            print(f"Step {step}, Loss {loss.item()}, Eval Loss {eval_loss}")


if __name__ == "__main__":
    args = get_args()
    dataset = ShakespeareDataset(block_size=args.block_size)
    model = get_model(args.model_type, args.block_size, dataset.vocab_size, args.n_layer, args.n_head, args.n_embd)
    train(model, dataset, args.num_steps, args.batch_size, args.learning_rate, args.weight_decay)
    torch.save(model.state_dict(), f"{args.model_type}_model.pt")
