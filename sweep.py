import wandb
import torch
from model import *
import torch.nn as nn
from load_data import *

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'embed_dim': {'values': [128, 256, 512]},
        'hidden_dim': {'values': [256, 512, 1024]},
        'cell_type': {'values': ['LSTM', 'GRU']},
        'enc_layers': {'values': [1, 2, 3]},
        'dec_layers': {'values': [1, 2]},
        'dropout': {'values': [0.3, 0.5]},
        'batch_size': {'values': [64]},
        'beam_size': {'values': [1, 3, 5]},
        'lr': {'min': 1e-4, 'max': 1e-3},
        'epochs': {'values': [10]}
    }
}


def train_sweep():
    wandb.init()
    config = wandb.config

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_loader, val_loader, _, dev_vocab, lat_vocab = get_data_loaders(
        batch_size=config.batch_size
    )
    
    # Model config
    class ModelConfig:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dev_vocab_size = len(dev_vocab)
        lat_vocab_size = len(lat_vocab)
        eos_token = lat_vocab['<EOS>']
        embed_dim = config.embed_dim
        hidden_dim = config.hidden_dim
        cell_type = config.cell_type
        enc_layers = config.enc_layers
        dec_layers = config.dec_layers
        dropout = config.dropout
    
    model = Transliterator(ModelConfig).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training phase
        for dev_input, lat_target in train_loader:
            dev_input = dev_input.to(device)
            lat_target = lat_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(dev_input, lat_target[:, :-1])  # Teacher forcing
            
            # Calculate loss
            loss = criterion(
                outputs.reshape(-1, ModelConfig.lat_vocab_size),
                lat_target[:, 1:].reshape(-1)
            )
            
            # Calculate accuracy
            preds = outputs.argmax(-1)
            targets = lat_target[:, 1:]
            mask = (targets != 0)
            correct = (preds == targets).float() * mask.float()
            
            train_correct += correct.sum().item()
            train_total += mask.sum().item()
            train_loss += loss.item()
            
            # Backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for dev_input, lat_target in val_loader:
                dev_input = dev_input.to(ModelConfig.device)
                lat_target = lat_target.to(ModelConfig.device)
                
                # Calculate validation loss
                outputs = model(dev_input, lat_target[:, :-1])
                loss = criterion(
                    outputs.reshape(-1, ModelConfig.lat_vocab_size),
                    lat_target[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
                
                # Calculate validation accuracy
                preds = outputs.argmax(-1)
                targets = lat_target[:, 1:]
                mask = (targets != 0)
                correct = (preds == targets).float() * mask.float()
                
                val_correct += correct.sum().item()
                val_total += mask.sum().item()
        
        # Calculate metrics
        train_loss_avg = train_loss / len(train_loader)
        train_acc = (train_correct / train_total) * 100 if train_total > 0 else 0
        val_loss_avg = val_loss / len(val_loader)
        val_acc = (val_correct / val_total) * 100 if val_total > 0 else 0
        
        # Format and print metrics
        print(f"Epoch {epoch+1:02d}/{config.epochs:02d} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Train Acc: {train_acc:05.2f}% | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Acc: {val_acc:05.2f}%")
        
        # Log to wandb
        wandb.log({
            'epoch': epoch+1,
            'train_loss': train_loss_avg,
            'train_acc': train_acc,
            'val_loss': val_loss_avg,
            'val_acc': val_acc
        })


if __name__ == '__main__':
    # Start sweep
    sweep_id = wandb.sweep(sweep_config, project="transliteration-sweep")
    wandb.agent(sweep_id, function=train_sweep, count=20)