import torch
import torch.nn as nn
import torch.nn.functional as F


def init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.15)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.15)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float) -> None:
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=-1) # nh: num head, hs: head size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        y: torch.Tensor = F.scaled_dot_product_attention(q, k, v,
                        attn_mask=None, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y)) ## (B, T, C)
        return y


class SelfAttnModel(nn.Module):
    def __init__(self, n_vocab: int,
                 n_embd: int,
                 max_length: int,
                 dropout: float) -> None:
        super(SelfAttnModel, self).__init__()
        self.max_length = max_length
        self.n_embd = n_embd
        self.emb = nn.Embedding(n_vocab, n_embd)
        self.pse = nn.Embedding(max_length, n_embd)
        self.csa = SelfAttention(n_embd, 4, 0.1)
        self.lin1 = nn.Linear(n_embd*max_length, n_embd*4)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(n_embd*4, n_embd)
        self.lin3 = nn.Linear(n_embd, 2)

        self.apply(init_)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T = x.size()
        x = self.emb(x) + self.pse(torch.arange(T, device=x.device))
        x = self.csa(x).view(B, -1)
        
        x = F.gelu(self.lin1(x))
        x = self.dropout(x)
        x = F.gelu(self.lin2(x))
        x = self.lin3(x)
        return x


class LinModel(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, max_length: int, dropout: float) -> None:
        super(LinModel, self).__init__()
        self.emb = nn.Embedding(n_vocab, n_embd)
        self.pse  = nn.Embedding(max_length, n_embd)
        self.lin1 = nn.Linear(n_embd*max_length, n_embd*4)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(n_embd*4, n_embd)
        self.lin3 = nn.Linear(n_embd, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T = x.size()
        x = self.emb(x) + self.pse(torch.arange(T, device=x.device))
        x = x.view(B, -1)
        x = F.gelu(self.lin1(x))
        x = self.dropout(x)
        x = F.gelu(self.lin2(x))
        x = self.lin3(x)
        return x
        