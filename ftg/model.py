"""
One Transformer, two attention modes.

  causal=True  -> each token sees only tokens to its left   (AR, AR-FIM)
  causal=False -> each token sees every token                (MDLM)

Modern "Llama-style" parts: pre-norm, RMSNorm, RoPE, SwiGLU, tied embeddings.
No time conditioning for MDLM (MDLM works without it; the mask count already tells the
model roughly how noisy the input is). Note this choice in the README.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 8000
    seq_len: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    ffn_mult: float = 8 / 3        # SwiGLU hidden size = d_model * 8/3 (about 1024)
    dropout: float = 0.0
    causal: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cache(seq_len, head_dim, base=10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len, device=device).float(), inv)  # (T, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd). Rotate pairs (x1, x2) by a position-dependent angle.
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd, self.causal = cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.causal
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # THE one-line difference between AR and diffusion:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(cfg.d_model * cfg.ffn_mult)
        hidden = 64 * math.ceil(hidden / 64)           # round up for GPU efficiency
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)   # gate
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=False)   # back down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ffn = RMSNorm(cfg.d_model), SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)   # pre-norm + residual
        return x + self.ffn(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight                     # tied embeddings
        cos, sin = rope_cache(cfg.seq_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():                   # GPT-2 style scaled init
            if n.endswith("out.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))                         # (B, T, vocab) logits

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.emb.weight.numel() if non_embedding else n