import torch
import torch.nn.functional as F
from torch.amp import autocast
from model import SelfAttnModel, LinModel
from utils import load_data
from string import ascii_lowercase
from tqdm import tqdm
from typing import Iterator


boy_names, girl_names = load_data()
blen, glen = len(boy_names), len(girl_names)
vocab = "# " + ascii_lowercase
n_vocab = 28
encode = lambda x: torch.tensor([vocab.index(c) for c in x], dtype=torch.long)

n_embd = 16
max_length = 11
max_iter = 2048*2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data() -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.zeros((blen+glen, max_length), dtype=torch.long)
    for i, name in enumerate(boy_names+girl_names):
        X[i, :len(name)] = encode(name)
    
    Y = torch.zeros((blen+glen,), dtype=torch.long)
    Y[blen:] = 1
    shf = torch.randperm(blen+glen)
    X = X[shf]
    Y = Y[shf]
    return X, Y


def loader(max_iter: int, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    X, Y = load_data()
    for i in range(max_iter):
        ix = torch.randint(0, blen+glen, (batch_size,))
        yield i, X[ix].to(device), Y[ix].to(device)


model = LinModel(n_vocab, n_embd, max_length, 0.1)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
ctx = autocast(device_type="cuda", dtype=torch.float16)
scaler = torch.cuda.amp.GradScaler(enabled=True)  

prog_bar = tqdm(loader(max_iter, 256), total=max_iter)
for i, xb, yb in prog_bar:
    with ctx:
        out = model(xb)
        loss = F.cross_entropy(out, yb)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    prog_bar.desc = f"{loss=:.4f}"


torch.save(model.state_dict(), "linmodel.pt")