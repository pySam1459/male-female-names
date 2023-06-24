# Inference script to predict the male-female classification of a name
import torch
from model import SelfAttnModel, LinModel
from string import ascii_lowercase

# '#' is the padding character
vocab = "# " + ascii_lowercase
n_vocab = 28 # 26 letters + ' ' + '#'
encode = lambda x: torch.tensor([vocab.index(c) for c in x], dtype=torch.long)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sd = torch.load("linmodel.pt") ## state_dict saved from training
n_embd = 16#64#4 ## approximate smallest embedding dimensions

# model = SelfAttnModel(n_vocab, n_embd, 11, 0.0)
model = LinModel(n_vocab, n_embd, 11, 0.0)
model.load_state_dict(sd)
model = model.to(device)
model.eval()

classes = ["male", "female"]
while True:
    name = input("Enter a name: ").lower()
    name += "#"*(11-len(name)) # add padding
    enc_name = encode(name).view(1, -1).to(device)

    logits: torch.Tensor = model(enc_name)
    class_ = classes[logits.argmax(dim=-1).item()]
    print(f"That name is {class_}")