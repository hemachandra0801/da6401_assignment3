import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from load_data import *
from attention_model import *

# Configuration
class Config:
    # Model parameters
    embed_dim = 256
    hidden_dim = 256
    num_layers = 2
    dropout = 0.2
    batch_size = 64
    lr = 0.0008705
    epochs = 10
    beam_size = 3
    enc_hid_dim = 2
    dec_hid_dim = 2
    cell_type = 'GRU'
    enc_layers = 2
    dec_layers = 2
    
    
    # Paths
    predictions_dir = 'predictions_attention'
    viz_filename = 'vanilla_attention.png'

def train_and_evaluate():
    # Create predictions directory
    os.makedirs(Config.predictions_dir, exist_ok=True)
    
    # Load data
    train_loader, dev_loader, test_loader, dev_vocab, latin_vocab = get_data_loaders(Config.batch_size)
    
    # Model config
    class ModelConfig:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lat_vocab_size = len(latin_vocab)
        dev_vocab_size = len(dev_vocab)
        embed_dim = Config.embed_dim
        enc_hid_dim = Config.enc_hid_dim
        dec_hid_dim = Config.dec_hid_dim
        cell_type = Config.cell_type
        enc_layers = Config.enc_layers
        dec_layers = Config.dec_layers
        dropout = Config.dropout
        hidden_dim = 512
        eos_token = latin_vocab['<EOS>']
        bidirectional = True
        attn_dim = 128
    
    # Initialize model
    model = AttnTransliterator(ModelConfig).to(ModelConfig.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Training loop
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        for src, trg in train_loader:
            src, trg = src.to(ModelConfig.device), trg.to(ModelConfig.device)
            optimizer.zero_grad()
            outputs = model(src, trg[:, :-1])
            loss = criterion(outputs.reshape(-1, ModelConfig.dev_vocab_size), 
                           trg[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}')

    # Evaluation
    model.eval()
    idx_to_dev = {v:k for k,v in dev_vocab.items()}
    idx_to_lat = {v:k for k,v in latin_vocab.items()}
    
    predictions = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (src, trg) in enumerate(test_loader):
            src, trg = src.to(ModelConfig.device), trg.to(ModelConfig.device)
            outputs = model(src, beam_size=Config.beam_size)
            
            # Process batch
            for i in range(src.size(0)):
                # Convert sequences to strings
                input_seq = [idx_to_lat[idx.item()] for idx in src[i] 
                             if idx.item() not in {0,1,2}]
                pred_seq = [idx_to_dev[idx.item()] for idx in outputs[i] 
                             if idx.item() not in {0,1,2}]
                target_seq = [idx_to_dev[idx.item()] for idx in trg[i] 
                             if idx.item() not in {0,1,2}]
                
                # Save to predictions
                predictions.append({
                    'input': ''.join(input_seq),
                    'prediction': ''.join(pred_seq),
                    'target': ''.join(target_seq),
                    'correct': pred_seq == target_seq
                })
                
                # Save to file
                with open(f'{Config.predictions_dir}/pred_{batch_idx}_{i}.txt', 'w') as f:
                    f.write(f"Input: {''.join(input_seq)}\n")
                    f.write(f"Prediction: {''.join(pred_seq)}\n")
                    f.write(f"Target: {''.join(target_seq)}\n")
                
                # Calculate accuracy
                if ''.join(pred_seq) == ''.join(target_seq):
                    correct += 1
                total += 1

    print(f'Test Accuracy: {correct/total:.2%}')

    # Save to CSV
    df = pd.DataFrame(predictions)
    os.makedirs(Config.predictions_dir, exist_ok=True)
    csv_path = os.path.join(Config.predictions_dir, 'predictions_vanilla.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # Visualization for 10 samples
    def visualize_predictions(samples):
        # font_path = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'
        # dev_font = fm.FontProperties(fname=font_path)
        
        fig, axs = plt.subplots(5, 2, figsize=(15, 25))
        axs = axs.flatten()
        
        for idx, sample in enumerate(samples[:10]):
            ax = axs[idx]
            text = (f"Input (Latin): {sample['input']}\n"
                    f"Prediction: {sample['prediction']}\n"
                    f"Target: {sample['target']}")
            
            ax.text(0.5, 0.5, text, 
                    ha='center', va='center', 
                    fontsize=12, family='Noto Sans Devanagari')
            ax.axis('off')
            ax.set_title(f'Sample {idx+1}', y=0.1)
        
        plt.tight_layout()
        plt.savefig(Config.viz_filename)
        plt.show()
    
    visualize_predictions(test_loader[:10])


train_and_evaluate()