import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

class TransliterationDataset(Dataset):
    def __init__(self, devanagari_samples, latin_samples, dev_vocab, latin_vocab):
        self.devanagari = devanagari_samples
        self.latin = latin_samples
        self.dev_vocab = dev_vocab
        self.latin_vocab = latin_vocab

    def __len__(self):
        return len(self.devanagari)

    def __getitem__(self, idx):
        dev_str = str(self.devanagari[idx])
        lat_str = str(self.latin[idx])
        
        dev_indices = [self.dev_vocab['<SOS>']] + [self.dev_vocab[c] for c in dev_str] + [self.dev_vocab['<EOS>']]
        lat_indices = [self.latin_vocab['<SOS>']] + [self.latin_vocab[c] for c in lat_str] + [self.latin_vocab['<EOS>']]
        
        return torch.tensor(dev_indices, dtype=torch.long), torch.tensor(lat_indices, dtype=torch.long)


def build_vocab(samples):
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
    for s in samples:
        for c in str(s):
            if c not in vocab:
                vocab[c] = len(vocab)
    return vocab

def load_data(lang_code="hi", data_dir="dakshina_dataset_v1.0"):
    base_path = os.path.join(data_dir, lang_code, "lexicons")
    
    # Load TSV files with proper column order (Devanagari first, then Latin)
    train_df = pd.read_csv(
        os.path.join(base_path, f"{lang_code}.translit.sampled.train.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"], 
        usecols=[0, 1], dtype=str
    )
    dev_df = pd.read_csv(
        os.path.join(base_path, f"{lang_code}.translit.sampled.dev.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"],
        usecols=[0, 1], dtype=str
    )
    test_df = pd.read_csv(
        os.path.join(base_path, f"{lang_code}.translit.sampled.test.tsv"),
        sep="\t", header=None, names=["devanagari", "latin", "count"],
        usecols=[0, 1], dtype=str
    )

    # Build vocabularies
    char_to_idx_devanagari = build_vocab(train_df["devanagari"].tolist())
    char_to_idx_latin = build_vocab(train_df["latin"].tolist())

    # Create datasets
    train_dataset = TransliterationDataset(
        train_df["devanagari"].tolist(),
        train_df["latin"].tolist(),
        char_to_idx_devanagari,
        char_to_idx_latin
    )
    dev_dataset = TransliterationDataset(
        dev_df["devanagari"].tolist(),
        dev_df["latin"].tolist(),
        char_to_idx_devanagari,
        char_to_idx_latin
    )
    test_dataset = TransliterationDataset(
        test_df["devanagari"].tolist(),
        test_df["latin"].tolist(),
        char_to_idx_devanagari,
        char_to_idx_latin
    )

    return train_dataset, dev_dataset, test_dataset, char_to_idx_devanagari, char_to_idx_latin

def collate_fn(batch):
    dev_batch, lat_batch = zip(*batch)
    dev_padded = torch.nn.utils.rnn.pad_sequence(dev_batch, batch_first=True, padding_value=0)
    lat_padded = torch.nn.utils.rnn.pad_sequence(lat_batch, batch_first=True, padding_value=0)
    return dev_padded, lat_padded

def get_data_loaders(batch_size=32, lang_code="hi"):
    train_dataset, dev_dataset, test_dataset, char_to_idx_devanagari, char_to_idx_latin = load_data(lang_code)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, dev_loader, test_loader, char_to_idx_devanagari, char_to_idx_latin

def print_samples(loader, num_samples=5, dev_vocab=None, lat_vocab=None):
    idx_to_dev = {v: k for k, v in dev_vocab.items()}
    idx_to_lat = {v: k for k, v in lat_vocab.items()}
    
    print("\nSample pairs from dataset (Devanagari → Latin):")
    for i, (dev_batch, lat_batch) in enumerate(loader):
        if i >= num_samples:
            break
            
        # Convert first sample in batch
        dev_seq = dev_batch[0].numpy()
        lat_seq = lat_batch[0].numpy()
        
        # Convert indices to characters
        dev_str = ''.join([idx_to_dev.get(idx, '?') for idx in dev_seq if idx not in {0, 1, 2}])
        lat_str = ''.join([idx_to_lat.get(idx, '?') for idx in lat_seq if idx not in {0, 1, 2}])
        
        print(f"{dev_str} -> {lat_str}")

if __name__ == "__main__":
    train_loader, dev_loader, test_loader, char_to_idx_devanagari, char_to_idx_latin = get_data_loaders()

    print(f"Devanagari vocab size: {len(char_to_idx_devanagari)}")
    print(f"Latin vocab size: {len(char_to_idx_latin)}")

    # Print sample pairs
    print_samples(train_loader, 5, char_to_idx_devanagari, char_to_idx_latin)

    # Example batch
    devanagari_batch, latin_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Devanagari: {devanagari_batch.shape}")  # (batch_size, max_seq_len)
    print(f"Latin: {latin_batch.shape}")