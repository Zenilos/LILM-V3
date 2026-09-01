from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 4388
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    ffn_hidden: int = 1024
    context_length: int = 256
