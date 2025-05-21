import wandb
import torch
from load_data import *
from attention_model import *

# Sweep Configuration
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'embed_dim': {'values': [128, 256, 512]},
        'enc_hid_dim': {'values': [256]},
        'dec_hid_dim': {'values': [256]},
        'attn_dim': {'values': [64, 128, 256]},
        'cell_type': {'values': ['LSTM', 'GRU']},
        'bidirectional': {'values': [True]},
        'enc_layers': {'values': [1, 2]},
        'dec_layers': {'values': [1, 2]},
        'dropout': {'values': [0.2, 0.5]},
        'batch_size': {'values': [64]},
        'lr': {'max': 0.001, 'min': 0.0001},
        'epochs': {'values': [3]}
    }
}


def train():
    wandb.init()
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, dev_loader, _, dev_vocab, latin_vocab = get_data_loaders(config.batch_size)
    
    class ModelConfig:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lat_vocab_size = len(latin_vocab)
        dev_vocab_size = len(dev_vocab)
        embed_dim = config.embed_dim
        enc_hid_dim = config.enc_hid_dim
        dec_hid_dim = config.dec_hid_dim
        attn_dim = config.attn_dim
        cell_type = config.cell_type
        bidirectional = config.bidirectional
        enc_layers = config.enc_layers
        dec_layers = config.dec_layers
        dropout = config.dropout
    
    model = AttnTransliterator(ModelConfig).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        # Training phase
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            
            # Forward pass with teacher forcing
            outputs = model(src, trg[:, :-1])  # Exclude last token from input
            
            # Calculate loss
            loss = criterion(
                outputs.reshape(-1, ModelConfig.dev_vocab_size),
                trg[:, 1:].reshape(-1)  # Exclude first token from target
            )
            
            # Calculate accuracy
            preds = outputs.argmax(-1)
            targets = trg[:, 1:]  # Exclude SOS token
            mask = (targets != 0)
            
            correct = (preds == targets).float() * mask.float()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate training metrics
        train_loss = total_loss / len(train_loader)
        train_acc = (total_correct / total_tokens) * 100 if total_tokens > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_tokens = 0
        
        with torch.no_grad():
            for src, trg in dev_loader:
                src, trg = src.to(device), trg.to(device)
                
                outputs = model(src, trg[:, :-1])
                
                # Calculate validation loss
                loss = criterion(
                    outputs.reshape(-1, ModelConfig.dev_vocab_size),
                    trg[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
                
                # Calculate validation accuracy
                preds = outputs.argmax(-1)
                targets = trg[:, 1:]
                mask = (targets != 0)
                
                correct = (preds == targets).float() * mask.float()
                val_correct += correct.sum().item()
                val_tokens += mask.sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(dev_loader)
        val_acc = (val_correct / val_tokens) * 100 if val_tokens > 0 else 0
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Print metrics
        print(f"Epoch {epoch+1:02d}/{config.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# Start the sweep
sweep_id = wandb.sweep(sweep_config, project="attention-sweep")
wandb.agent(sweep_id, function=train, count=20)