import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        combined = torch.cat((decoder_hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(combined))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class AttnTransliterator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder (Input: Latin)
        self.encoder_embed = nn.Embedding(config.lat_vocab_size, config.embed_dim)
        self.encoder_rnn = getattr(nn, config.cell_type)(
            input_size=config.embed_dim,
            hidden_size=config.enc_hid_dim,
            num_layers=config.enc_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=config.dropout if config.enc_layers > 1 else 0
        )
        
        # Hidden state projection (Enc → Dec dimensions)
        self.hidden_proj = nn.Linear(
            config.enc_hid_dim * (2 if config.bidirectional else 1) * config.enc_layers,
            config.dec_hid_dim * config.dec_layers
        )
        
        # Decoder (Output: Devanagari)
        self.decoder_embed = nn.Embedding(config.dev_vocab_size, config.embed_dim)
        self.attention = Attention(
            enc_hid_dim=config.enc_hid_dim * (2 if config.bidirectional else 1),
            dec_hid_dim=config.dec_hid_dim,
            attn_dim=config.attn_dim
        )
        self.decoder_rnn = getattr(nn, config.cell_type)(
            input_size=config.embed_dim + (config.enc_hid_dim * (2 if config.bidirectional else 1)),
            hidden_size=config.dec_hid_dim,
            num_layers=config.dec_layers,
            batch_first=True,
            dropout=config.dropout if config.dec_layers > 1 else 0
        )
        self.fc = nn.Linear(config.dec_hid_dim + (config.enc_hid_dim * (2 if config.bidirectional else 1)), 
                     config.dev_vocab_size)

    def encode(self, src):
        embedded = self.encoder_embed(src)
        outputs, hidden = self.encoder_rnn(embedded)
        return outputs, self._adapt_hidden(hidden)

    def forward(self, src, trg=None, beam_size=1):
        encoder_outputs, hidden = self.encode(src)
        if trg is not None:
            return self._teacher_force(encoder_outputs, hidden, trg)
        else:
            return self._beam_search(encoder_outputs, hidden, beam_size)

    def _adapt_hidden(self, hidden):
        if isinstance(hidden, tuple):
            # LSTM: (h, c)
            h, c = hidden
            batch_size = h.size(1)
            
            # Flatten and project
            h_flat = h.permute(1, 0, 2).contiguous().view(batch_size, -1)
            c_flat = c.permute(1, 0, 2).contiguous().view(batch_size, -1)
            
            h_proj = self.hidden_proj(h_flat)
            c_proj = self.hidden_proj(c_flat)
            
            # Reshape to (num_layers, batch_size, dec_hid_dim)
            h_proj = h_proj.view(
                batch_size, self.config.dec_layers, self.config.dec_hid_dim
            ).permute(1, 0, 2).contiguous()
            
            c_proj = c_proj.view(
                batch_size, self.config.dec_layers, self.config.dec_hid_dim
            ).permute(1, 0, 2).contiguous()
            
            return (h_proj, c_proj)
        else:
            # GRU/RNN
            batch_size = hidden.size(1)
            hidden_flat = hidden.permute(1, 0, 2).contiguous().view(batch_size, -1)
            hidden_proj = self.hidden_proj(hidden_flat)
            hidden_proj = hidden_proj.view(
                batch_size, self.config.dec_layers, self.config.dec_hid_dim
            ).permute(1, 0, 2).contiguous()
            return hidden_proj

    def _teacher_force(self, encoder_outputs, hidden, trg):
        batch_size, trg_len = trg.size(0), trg.size(1)
        outputs = torch.zeros(batch_size, trg_len, self.config.dev_vocab_size).to(self.config.device)
        input = trg[:, 0]  # SOS
        
        for t in range(1, trg_len):
            output, hidden = self._decode_step(input.unsqueeze(1), hidden, encoder_outputs)
            outputs[:, t] = output
            input = trg[:, t]
            
        return outputs

    def _decode_step(self, input, hidden, encoder_outputs):
        embedded = self.decoder_embed(input)
        
        # Get decoder hidden state for attention
        if isinstance(hidden, tuple):
            decoder_hidden = hidden[0][-1]
        else:
            decoder_hidden = hidden[-1]
        
        # Compute attention
        attn_weights = self.attention(decoder_hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        
        # Combine with embedded input
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # RNN step
        output, hidden = self.decoder_rnn(rnn_input, hidden)
        
        # Final prediction
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        return self.fc(output), hidden