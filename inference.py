import torch
from model import SelfAttnModel, LinModel
from string import ascii_lowercase

vocab = "# " + ascii_lowercase
n_vocab = 28
encode = lambda x: torch.tensor([vocab.index(c) for c in x], dtype=torch.long)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load("linmodel.pt")
n_embd = 16#64#4

# model = SelfAttnModel(n_vocab, n_embd, 11, 0.0)
model = LinModel(n_vocab, n_embd, 11, 0.0)
model.load_state_dict(sd)
model = model.to(device)
model.eval()

classes = ["male", "female"]
while True:
    name = input("Enter a name: ").lower()
    name += "#"*(11-len(name))
    enc_name = encode(name).view(1, -1).to(device)
    out: torch.Tensor = model(enc_name)
    print(f"That name is {classes[out.argmax().item()]}")