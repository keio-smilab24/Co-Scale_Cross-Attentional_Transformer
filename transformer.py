import torch
import torch.nn as nn
import copy


class MultiHeadAttention(nn.Module):
    """Input: tensor in shape [B, C, H, W]"""
    def __init__(self, dim, num_heads=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.dp1 = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.dp2 = nn.Dropout(proj_drop)

    def forward(self, k, q, v):
        b, c, h, w = k.shape
        n = h * w
        k = torch.flatten(k, start_dim=2).transpose(-2, -1)
        q = torch.flatten(q, start_dim=2).transpose(-2, -1)
        v = torch.flatten(v, start_dim=2).transpose(-2, -1)

        k = k.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.dp1(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.dp2(out)
        out = out.transpose(-2, -1).reshape(b, c, h, w)

        return out
