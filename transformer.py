import torch
from tokenizers import Tokenizer
from typing import Tuple

from layers import Encoder, Decoder, EncoderLayer, DecoderLayer, MultiheadAttention, FeedForward, Embedding

def get_pad_mask(x: torch.Tensor, pad_index: int):
    return (x != pad_index).unsqueeze(-2)

def get_subsequent_mask(x: torch.Tensor):
    batch_size, seq_len = x.size()
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask

class Transformer(torch.nn.Module):
    def __init__(
            self, 
            d_model: int = 256, 
            num_heads: int = 8, 
            d_ff: int = 512, 
            num_layers: int = 6, 
            vocab_size: int = 1000, 
            pad_index: int = 1, 
            dropout: float = 0.1,
            max_len: int = 64,
            tokenizer: Tokenizer = None,
            device: str = 'cuda',
        ):
        super().__init__()
        mha = MultiheadAttention(d_model, num_heads)
        enc_dec_mha = MultiheadAttention(d_model, num_heads)
        ffn = FeedForward(d_model, d_ff)
        self.encoder = Encoder(EncoderLayer(mha, ffn, dropout), num_layers)
        self.decoder = Decoder(DecoderLayer(mha, enc_dec_mha, ffn, dropout), num_layers)
        self.normalize = torch.nn.LayerNorm(d_model)

        self.src_embedding = Embedding(d_model, vocab_size, pad_index)
        self.tgt_embedding = Embedding(d_model, vocab_size, pad_index)
        self.vocab_projection = torch.nn.Linear(d_model, vocab_size)

        self.pad_index= pad_index
        self.device = device
        self.max_len = max_len
        self.tokenizer = tokenizer
        
    def predict(self, x: torch.Tensor):
        memory, mask = self.encode_src(x)
        start_idx = self.tokenizer.token_to_id('<s>')
        end_idx = self.tokenizer.token_to_id('</s>')
        pad_idx = self.tokenizer.token_to_id('<pad>')
        batch_size = memory.size(0)
        max_len = self.max_len

        trg_tensor = torch.full(
            (batch_size, max_len),
            pad_idx,
            device=x.device,
            dtype=torch.long
        )
        trg_tensor[:, 0] = start_idx

        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        for step in range(1, max_len):
            if finished.all():
                break
            active_indices = torch.nonzero(~finished).view(-1)
            active_memory = memory[active_indices]
            active_trg = trg_tensor[active_indices, :step]
            active_mask = mask[active_indices]
            output = self.decode_tgt(active_trg, active_memory, active_mask)
            current_logits = output[:, -1, :]

            next_tokens = current_logits.argmax(dim=-1)
            trg_tensor[active_indices, step] = next_tokens

            finished[active_indices] |= (next_tokens == end_idx)
        return trg_tensor

    def encode_src(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_mask = get_pad_mask(x, self.pad_index).to(self.device)
        x = self.src_embedding(x)
        return self.encoder.forward(x, pad_mask), pad_mask
    
    def decode_tgt(self, x, memory, src_mask) -> torch.Tensor:
        tgt_mask = get_pad_mask(x, self.pad_index) & get_subsequent_mask(x).to(self.device)
        x = self.tgt_embedding(x)
        x = self.decoder.forward(x, memory, src_mask, tgt_mask)
        x = self.normalize(x)
        return self.vocab_projection(x)
    
    def forward(self, x, y):
        memory, mask = self.encode_src(x)
        out = self.decode_tgt(y, memory, mask)
        return out
    