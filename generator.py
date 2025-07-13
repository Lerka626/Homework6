import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
from tokenizers import Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json

CONFIG = {
    'data_path': 'data.txt', 
    'tokenizer_path': 'tokenizer.json', 
    'd_model': 128, 
    'n_layers': 4,
    'n_heads': 4,
    'd_ff': 512,
    'max_length': 64,
    'dropout': 0.1,
    'lr': 1e-4,
    'batch_size': 1,
    'num_epochs': 3, 
    'save_dir': 'results',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu' 
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# Класс позиционного кодирования
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Класс блока декодера
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        return self.norm2(x)

class GeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, n_layers, n_heads, d_ff, max_length, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_length = max_length
        self.pad_idx = pad_idx

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(x.device)
        
        x = self.token_embed(x)
        x = self.pos_encoding(x)
        for block in self.decoder_blocks:
            x = block(x, mask)
        return self.fc(x)
    
    def generate(self, prompt, tokenizer, context_len=50, temperature=1.0, max_out_tokens=200):
        """Авторегрессивная генерация текста"""
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            encoding = tokenizer.encode(prompt)
            input_ids = encoding.ids
            input_ids = torch.tensor([input_ids]).to(device)
            generated = input_ids.clone()
            
            eos_token_id = tokenizer.token_to_id('</s>')
            if eos_token_id is None:
                eos_token_id = tokenizer.token_to_id('[EOS]') or tokenizer.token_to_id('<|endoftext|>') or 2
            
            for _ in range(max_out_tokens):
                context = generated[:, -context_len:]
                logits = self(context)[:, -1, :]
                
                # Защита от нулевой температуры
                if temperature <= 0:
                    temperature = 0.01
                scaled_logits = logits / temperature
                
                # Обработка NaN/Inf
                if torch.isnan(scaled_logits).any() or torch.isinf(scaled_logits).any():
                    scaled_logits = torch.nan_to_num(scaled_logits, nan=0.0, posinf=1e4, neginf=-1e4)
                
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # Проверка на корректность вероятностей
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == eos_token_id:
                    break
            
            try:
                decoded = tokenizer.decode(generated[0].tolist())
            except Exception as e:
                print(f"Ошибка декодирования: {e}")
                decoded = prompt + " [ОШИБКА ГЕНЕРАЦИИ]"
            
            return decoded

def prepare_data(text, tokenizer, max_length, step_size):
    """Создание датасета из текста"""
    encoding = tokenizer.encode(text)
    tokens = encoding.ids
    sequences = []
    
    for i in range(0, len(tokens), step_size):
        chunk = tokens[i:i + max_length]
        if len(chunk) < max_length // 2:
            continue
        sequences.append(torch.tensor(chunk))
    
    return sequences

def collate_fn(batch, pad_token_id):
    """Обработка батча с паддингом"""
    padded = torch.nn.utils.rnn.pad_sequence(
        batch, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    inputs = padded[:, :-1]
    targets = padded[:, 1:]
    return inputs, targets

def create_simple_tokenizer(text, save_path):
    """Создание простого токенизатора на базе словаря"""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder()
    
    from tokenizers.trainers import BpeTrainer
    trainer = BpeTrainer(
        special_tokens=["[PAD]", "[UNK]", "<s>", "</s>"],
        min_frequency=2
    )
    
    with open("temp.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    tokenizer.train(files=["temp.txt"], trainer=trainer)
    
    tokenizer.save(save_path)
    os.remove("temp.txt")
    return tokenizer

def train_model():
    data_path = CONFIG['data_path']
    if not os.path.exists(data_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, 'data.txt')
        
        if not os.path.exists(data_path):
            print(f"Файл не найден: {CONFIG['data_path']}")
            print(f"Текущая директория: {os.getcwd()}")
            print(f"Список файлов: {os.listdir()}")
            raise FileNotFoundError(f"Файл данных не найден: {CONFIG['data_path']}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text_data = f.read()

    if not os.path.exists(CONFIG['tokenizer_path']):
        print("Создаем токенизатор...")
        tokenizer = create_simple_tokenizer(text_data, CONFIG['tokenizer_path'])
    else:
        tokenizer = Tokenizer.from_file(CONFIG['tokenizer_path'])
    
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    print(f"Размер словаря: {vocab_size}")
    
    special_tokens = ["[PAD]", "[UNK]", "<s>", "</s>"]
    for token in special_tokens:
        if tokenizer.token_to_id(token) is None:
            print(f"Внимание: токен '{token}' отсутствует в словаре!")
    
    sequences = prepare_data(
        text_data, 
        tokenizer,
        max_length=CONFIG['max_length'],
        step_size=CONFIG['max_length'] // 2
    )
    
    dataloader = torch.utils.data.DataLoader(
        sequences,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id)
    )
    
    model = GeneratorTransformer(
        vocab_size=vocab_size,
        pad_idx=pad_token_id,
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        n_heads=CONFIG['n_heads'],
        d_ff=CONFIG['d_ff'],
        max_length=CONFIG['max_length'],
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    losses = []
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
            inputs, targets = inputs.to(CONFIG['device']), targets.to(CONFIG['device'])
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            if torch.isnan(logits).any():
                print("Обнаружены NaN в логитах! Пропускаем батч.")
                continue
                
            loss = criterion(
                logits.view(-1, logits.size(-1)), 
                targets.contiguous().view(-1)
            )
            if torch.isnan(loss):
                print("Обнаружены NaN в loss! Пропускаем батч.")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        checkpoint_path = os.path.join(CONFIG['save_dir'], f'epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, CONFIG['num_epochs']+1), losses, 'o-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(CONFIG['save_dir'], 'training_loss.png'))
    print("Saved loss plot")
    
    return model, tokenizer

def test_model(model, tokenizer):
    """Интерактивный чат с моделью"""
    print("Бот готов к общению! Введите 'quit' для выхода")
    while True:
        user_input = input("Вы: ")
        if user_input.lower() == 'quit':
            break
            
        try:
            response = model.generate(
                prompt=user_input,
                tokenizer=tokenizer,
                context_len=min(32, CONFIG['max_length'] // 2), 
                temperature=0.8,
                max_out_tokens=50 
            )
            print(f"Бот: {response}")
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            print("Бот: Извините, произошла ошибка. Попробуйте еще раз.")

if __name__ == '__main__':
    model, tokenizer = train_model()
    
    test_model(model, tokenizer)
