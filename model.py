import torch
import torch.nn as nn
from config import *

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_dim = EMBED_DIM // NUM_HEADS

        self.qkv = nn.Linear(EMBED_DIM, EMBED_DIM * 3)
        self.proj = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, NUM_HEADS, self.head_dim).transpose(1,2)
        k = k.view(B, T, NUM_HEADS, self.head_dim).transpose(1,2)
        v = v.view(B, T, NUM_HEADS, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2,-1)) / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T,T)).to(x.device)
        att = att.masked_fill(mask==0, float('-inf'))

        att = torch.softmax(att, dim=-1)

        out = att @ v

        out = out.transpose(1,2).contiguous().view(B,T,C)

        return self.proj(out)


class Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = Attention()

        self.ln2 = nn.LayerNorm(EMBED_DIM)

        self.ff = nn.Sequential(
            nn.Linear(EMBED_DIM, FFN_DIM),
            nn.GELU(),
            nn.Linear(FFN_DIM, EMBED_DIM)
        )

    def forward(self, x):

        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x


class SahilGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, EMBED_DIM)

        self.blocks = nn.Sequential(
            *[Block() for _ in range(NUM_LAYERS)]
        )

        self.ln = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):

        B,T = x.shape

        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))

        x = tok + pos

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.head(x)

        return logits