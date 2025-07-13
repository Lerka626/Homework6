import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from transformer import Transformer
from wmt_dataset import create_dataloaders, WMTDataset


class TransformerTrainer:
    def __init__(self, model, train_loader, val_loader, dataset, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset: WMTDataset = dataset
        self.config = config
        self.device = device
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=dataset.get_pad_token_id())
        self.optimizer = optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.best_bleu = 0.0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        train_bleu = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        scaler = GradScaler()
        for i, batch in enumerate(progress_bar):
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            
            self.optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                logits = self.model.forward(src_ids, tgt_ids[:, :-1])
            
                targets = tgt_ids[:, 1:]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                scaler.scale(loss).backward()

                for param in self.model.parameters():
                    if torch.isnan(param.grad).any():
                        param.grad = 0

                train_bleu += self.compute_bleu(logits.argmax(dim=-1), targets)

            scaler.step(self.optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{(total_loss / (i + 1)):.4f}', 'bleu': f'{(train_bleu / (i + 1)):.4f}'})
        
        return {'loss': total_loss / (i + 1), 'bleu': train_bleu / (i + 1)}
    
    def validate(self, epoch):
        self.model.eval()
        bleu_score = 0.0
        
        for i, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
            src_ids = batch['src_ids'].to(self.device)
            tgt_ids = batch['tgt_ids'].to(self.device)
            
            targets = tgt_ids[:, 1:]
            
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
                generated_ids = self.model.predict(src_ids)

            bleu_score += self.compute_bleu(generated_ids, targets)

        bleu_score /= (i + 1)
        
        if bleu_score > self.best_bleu:
            self.best_bleu = bleu_score
            self.save_checkpoint('best_model.pt')
        
        return {
            'bleu': bleu_score
        }
    
    def compute_bleu(self, pred: torch.Tensor, true: torch.Tensor):
        pred_texts = self.dataset.tokenizer.decode_batch(pred.cpu().numpy())
        true_texts = self.dataset.tokenizer.decode_batch(true.cpu().numpy())
        bleu_score = 0.0
        for p, t in zip(pred_texts, true_texts):
            bleu_score += sentence_bleu([t.split()], p.split(), smoothing_function=SmoothingFunction().method4, auto_reweigh=True)
        return bleu_score / len(pred_texts)
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_bleu': self.best_bleu,
        }
        
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        print(f'Checkpoint saved: {path}')
    
    def train(self):
        num_epochs = self.config.get('num_epochs', 100)
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            train_metrics = self.train_epoch(epoch)
            print(f'Train - Loss: {train_metrics["loss"]:.4f}, BLEU: {train_metrics["bleu"]:.4f}')
            
            val_metrics = self.validate(epoch)
            print(f'Val - BLEU: {val_metrics["bleu"]:.4f}')
            
            self.scheduler.step()
            
            if (epoch + 1) % self.config.get('save_epochs', 5) == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}.pt')


def main():
    config = {
        'batch_size': 256,
        'max_length': 64,
        'max_train_samples': 5000000,
        'max_val_samples': 3000,
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'd_ff': 1024,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'save_epochs': 5,
        'save_dir': 'checkpoints',
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_loader, val_loader, dataset = create_dataloaders(
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        max_train_samples=config['max_train_samples'],
        max_val_samples=config['max_val_samples'],
    )
    
    model = Transformer(
        d_model=config['d_model'],
        num_heads=config['nhead'],
        d_ff=config['d_ff'],
        num_layers=config['num_encoder_layers'],
        vocab_size=dataset.get_vocab_size(),
        pad_index=dataset.get_pad_token_id(),
        dropout=config['dropout'],
        tokenizer=dataset.tokenizer,
        max_len=config['max_length'],
    )
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    trainer = TransformerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=dataset,
        config=config,
        device=device,
    )
    
    trainer.train()


if __name__ == '__main__':
    main() 