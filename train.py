import torch
import torch.nn.functional as F
from torch.amp import autocast
import requests as r
from bs4 import BeautifulSoup
from model import SelfAttnModel, LinModel
from string import ascii_lowercase
from tqdm import tqdm
from typing import Iterator


def load_data() -> tuple[list[str], list[str]]:
    """Load data from baby name websites"""
    mURL = "https://www.pampers.co.uk/pregnancy/baby-names/article/top-baby-names-for-boys"
    html = r.get(mURL)
    soup = BeautifulSoup(html.content, 'html.parser')
    names_htmls = soup.find_all("ol")
    boy_names_html = []
    for names_html in names_htmls:
        boy_names_html += names_html.find_all("p", {"class": "rich-text-text"})
    boy_names = [tag.text.strip("\t").lower() for tag in boy_names_html]

    fURL = "https://www.goodhousekeeping.com/life/parenting/a37668901/top-baby-girl-names/"
    html = r.get(fURL)
    soup = BeautifulSoup(html.content, 'html.parser')
    names_html = soup.find("ol", {"class": "css-1rk79nl et3p2gv0", "data-node-id": "34"})
    girl_names_html = names_html.find_all("li")
    girl_names = [tag.text.lower() for tag in girl_names_html]
    return boy_names, girl_names


boy_names, girl_names = load_data()
blen, glen = len(boy_names), len(girl_names)
vocab = "# " + ascii_lowercase
n_vocab = 28
encode = lambda x: torch.tensor([vocab.index(c) for c in x], dtype=torch.long)
# not decode is required as we predict class directly

n_embd = 16 ## 16 is approximately lowest embedding dimension to product good results
max_length = 11
max_iter = 2048*2
batch_size = 256
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

## to use mixed precision training
ctx = autocast(device_type="cuda", dtype=torch.float16)
scaler = torch.cuda.amp.GradScaler(enabled=True)  

prog_bar = tqdm(loader(max_iter, batch_size), total=max_iter)
for i, xb, yb in prog_bar: # iterate over training data
    with ctx:
        out = model(xb)
        loss = F.cross_entropy(out, yb)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    prog_bar.desc = f"{loss=:.4f}"

## save model state_dict
torch.save(model.state_dict(), "linmodel.pt")