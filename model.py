"""
Small transformer for next-token prediction on Mess3 sequences.

Architecture: GPT-2 style (pre-norm) with causal self-attention.
Supports activation storage for residual stream analysis.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 3
    context_length: int = 12
    d_model: int = 64
    n_heads: int = 4
    d_mlp: int = 256
    n_layers: int = 1
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        # Causal mask
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.context_length, config.context_length))
            .view(1, 1, config.context_length, config.context_length)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out, attn_weights


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_mlp)
        self.fc2 = nn.Linear(config.d_mlp, config.d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x, store_activations=False):
        activations = {}

        # Pre-norm attention + residual
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        if store_activations:
            activations['post_attn'] = x.detach()
            activations['attn_weights'] = attn_weights.detach()

        # Pre-norm MLP + residual
        x = x + self.mlp(self.ln2(x))
        if store_activations:
            activations['post_mlp'] = x.detach()

        return x, activations


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.context_length, config.d_model)
        self.block = TransformerBlock(config)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.unembed = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.store_activations = False
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, T) int tensor

        Returns:
            logits: (B, T, vocab_size)
            activations: dict (only populated if self.store_activations is True)
        """
        B, T = tokens.shape
        activations = {}

        pos = torch.arange(T, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(pos)

        if self.store_activations:
            activations['post_embed'] = x.detach()

        x, block_acts = self.block(x, store_activations=self.store_activations)
        activations.update(block_acts)

        x = self.ln_f(x)
        if self.store_activations:
            activations['post_ln'] = x.detach()

        logits = self.unembed(x)

        return logits, activations

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def verify_model():
    """Run sanity checks on the model."""
    config = ModelConfig()
    model = Transformer(config)

    n_params = model.count_parameters()
    print(f"Parameter count: {n_params:,}")

    # Check output shapes
    tokens = torch.randint(0, 3, (2, config.context_length))
    model.store_activations = True
    logits, acts = model(tokens)

    assert logits.shape == (2, config.context_length, 3), f"Logits shape: {logits.shape}"
    assert acts['post_embed'].shape == (2, config.context_length, config.d_model)
    assert acts['post_attn'].shape == (2, config.context_length, config.d_model)
    assert acts['post_mlp'].shape == (2, config.context_length, config.d_model)
    assert acts['post_ln'].shape == (2, config.context_length, config.d_model)
    assert acts['attn_weights'].shape == (2, config.n_heads, config.context_length, config.context_length)

    # Causal mask: attention weights should be zero for future positions
    attn = acts['attn_weights'][0, 0]  # (T, T)
    for i in range(config.context_length):
        assert torch.allclose(attn[i, i+1:], torch.zeros(config.context_length - i - 1)), \
            f"Position {i} attends to future"

    # Attention weights should sum to 1
    attn_sums = acts['attn_weights'].sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), \
        "Attention doesn't sum to 1"

    print("All model checks passed.")
    print(f"Shapes: logits={logits.shape}, embed={acts['post_embed'].shape}, "
          f"attn_weights={acts['attn_weights'].shape}")


if __name__ == '__main__':
    verify_model()
