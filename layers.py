import torch
import math
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Callable, Any
import os
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(2 * torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class Embedding(torch.nn.Module):
    def __init__(self, d_model: int, vocab_len: int, pad_index: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_len, self.d_model, padding_idx=pad_index)
        self.positional_encoding = PositionalEncoding(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return x
    
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.factor = math.sqrt(self.d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        attention_weights = torch.nn.Softmax(dim=-1)(scores)
        out = torch.matmul(attention_weights, value)
        return out, attention_weights
    
class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_linear = torch.nn.Linear(d_model, d_model)
        self.k_linear = torch.nn.Linear(d_model, d_model)
        self.v_linear = torch.nn.Linear(d_model, d_model)
        self.out_linear = torch.nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        query = self.q_linear(query)
        query = query.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        key = self.k_linear(key)
        key = key.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        value = self.v_linear(value)
        value = value.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attention_weights = self.attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        x = self.out_linear(x)
        x = self.dropout(x)
        return x, attention_weights
    
class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if d_ff % d_model:
            raise ValueError(f"Feed forward dimension {d_ff} must be divisible by model dimension {d_model}")
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.linear2(self.dropout(x))
        return x
    
class EncoderLayer(torch.nn.Module):
    def __init__(self, mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = deepcopy(mha)
        self.ffn = deepcopy(ffn)
        self.layernorm1 = torch.nn.LayerNorm(mha.d_model)
        self.layernorm2 = torch.nn.LayerNorm(mha.d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.layernorm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm, mask)[0]
        x_norm = self.layernorm2(x)
        x = self.dropout(x + self.ffn(x_norm))
        return x
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, mha: MultiheadAttention, enc_dec_mha: MultiheadAttention, ffn: FeedForward, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention1 = deepcopy(mha)
        self.attention2 = deepcopy(enc_dec_mha)
        self.ffn = deepcopy(ffn)
        self.layernorm1 = torch.nn.LayerNorm(mha.d_model)
        self.layernorm2 = torch.nn.LayerNorm(mha.d_model)
        self.layernorm3 = torch.nn.LayerNorm(mha.d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x_norm = self.layernorm1(x)
        x = x + self.attention1(x_norm, x_norm, x_norm, tgt_mask)[0]
        
        # Encoder-decoder attention с маской энкодера
        x_norm = self.layernorm2(x)
        x = x + self.attention2(x_norm, encoder_memory, encoder_memory, src_mask)[0]
        
        # Feed-forward с маской декодера
        x_norm = self.layernorm3(x)
        x = self.dropout(x + self.ffn(x_norm))
        return x
    
class Encoder(torch.nn.Module):
    def __init__(self, enc_layer: EncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([deepcopy(enc_layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, dec_layer: DecoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([deepcopy(dec_layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, encoder_memory: torch.Tensor, src_mask: Optional[torch.Tensor], tgt_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_memory, src_mask, tgt_mask)
        return x
    