import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer
import os
from tqdm import tqdm


class WMTDataset(Dataset):
    """WMT14 dataset wrapper for machine translation."""
    
    def __init__(self, split="train", 
                 max_length=128, max_samples=None):
        self.max_length = max_length
        self.max_samples = max_samples

        self.src_lang = "de"
        self.tgt_lang = "en"
        self.tokenizer: Tokenizer = Tokenizer.from_file(f"{os.path.dirname(__file__)}/mistral_tokenizer.json")
        self.tokenizer.add_special_tokens(['<pad>', '<s>', '</s>'])
        
        self.dataset = load_dataset("wmt14", f'{self.src_lang}-{self.tgt_lang}')[split]
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        def filter_function(example):
            translation = example["translation"]
            if self.src_lang in translation and self.tgt_lang in translation:
                src_ids = self.tokenizer.encode(translation[self.src_lang]).ids
                tgt_ids = self.tokenizer.encode(translation[self.tgt_lang]).ids
                return (len(src_ids) + 2) <= self.max_length and (len(tgt_ids) + 2) <= self.max_length
            return False

        valid_indices = []
        for i in tqdm(range(len(self.dataset)), desc='Filtering dataset'):
            if filter_function(self.dataset[i]):
                valid_indices.append(i)
        
        self.dataset = self.dataset.select(valid_indices)
        print(f'Filtered dataset to {len(self.dataset)} samples')

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_pad_token_id(self):
        return self.tokenizer.token_to_id('<pad>')
    
    def get_bos_token_id(self):
        return self.tokenizer.token_to_id('<s>')
    
    def get_eos_token_id(self):
        return self.tokenizer.token_to_id('</s>')
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        bos_token_id = self.tokenizer.token_to_id('<s>')
        eos_token_id = self.tokenizer.token_to_id('</s>')
        pad_token_id = self.tokenizer.token_to_id('<pad>')
        
        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        src_encoding = [bos_token_id] + self.tokenizer.encode(src_text).ids + [eos_token_id]
        tgt_encoding = [bos_token_id] + self.tokenizer.encode(tgt_text).ids + [eos_token_id]

        if len(src_encoding) > self.max_length or len(tgt_encoding) > self.max_length:
            raise ValueError(f'Source or target encoding is too long: {len(src_encoding)} or {len(tgt_encoding)}')
        
        return {
            'src_ids': torch.tensor(src_encoding),
            'tgt_ids': torch.tensor(tgt_encoding)
        }

class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        src_ids = [item['src_ids'] for item in batch if item is not None]
        tgt_ids = [item['tgt_ids'] for item in batch if item is not None]
        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=self.pad_token_id)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=self.pad_token_id)
        return {
            'src_ids': src_ids,
            'tgt_ids': tgt_ids
        }


def create_dataloaders(
    batch_size=32, 
    max_length=128, 
    max_train_samples=None, 
    max_val_samples=None, 
    num_workers=0
):  
    train_dataset = WMTDataset(
        split="train",
        max_length=max_length, 
        max_samples=max_train_samples
    )
    
    val_dataset = WMTDataset(
        split="validation",
        max_length=max_length, 
        max_samples=max_val_samples
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=Collator(train_dataset.tokenizer.token_to_id('<pad>')),
        num_workers=num_workers, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=Collator(val_dataset.tokenizer.token_to_id('<pad>')),
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset 
