import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

class Beam:
    def __init__(self, width, device, eos_token):
        self.width = width
        self.device = device
        self.eos_token = eos_token
        self.done = False
        self.hypotheses = [{
            'tokens': [1],  # SOS
            'score': 0.0,
            'hidden': None,
            'done': False
        }]

    def update(self, log_probs, hidden):
        new_hyps = []
        for hyp_idx, hyp in enumerate(self.hypotheses):
            if hyp['done']:
                new_hyps.append(hyp)
                continue
                
            # Get topk for current hypothesis
            scores, indices = log_probs[hyp_idx].topk(self.width)
            for i in range(self.width):
                token = indices[i].item()  # Now scalar
                score = hyp['score'] + scores[i].item()
                new_tokens = hyp['tokens'] + [token]
                done = (token == self.eos_token) or (len(new_tokens) >= 50)
                
                # Handle hidden state selection
                if isinstance(hidden, tuple):  # LSTM
                    new_h = (hidden[0][:, hyp_idx:hyp_idx+1, :],
                           hidden[1][:, hyp_idx:hyp_idx+1, :])
                else:  # GRU/RNN
                    new_h = hidden[:, hyp_idx:hyp_idx+1, :]
                
                new_hyps.append({
                    'tokens': new_tokens,
                    'score': score,
                    'hidden': new_h,
                    'done': done
                })
        
        # Sort and keep top-k hypotheses
        new_hyps.sort(key=lambda x: x['score']/(len(x['tokens']))**0.7, reverse=True)
        self.hypotheses = new_hyps[:self.width]
        self.done = all(h['done'] for h in self.hypotheses)
        return self.done

    def _select_hidden(self, hidden, idx):
        if isinstance(hidden, tuple):
            return (hidden[0][:, idx:idx+1, :],
                    hidden[1][:, idx:idx+1, :])
        return hidden[:, idx:idx+1, :]

    def get_current_tokens(self):
        return [hyp['tokens'][-1] for hyp in self.hypotheses]

    def get_current_hidden(self):
        if isinstance(self.hypotheses[0]['hidden'], tuple):
            # For LSTM
            h = torch.cat([h['hidden'][0] for h in self.hypotheses], dim=1)
            c = torch.cat([h['hidden'][1] for h in self.hypotheses], dim=1)
            return (h, c)
        else:
            # For GRU/RNN
            return torch.cat([h['hidden'] for h in self.hypotheses], dim=1)

    def get_best_sequence(self):
        best_hyp = max(self.hypotheses, key=lambda x: x['score'])
        return torch.tensor(best_hyp['tokens'][1:-1], device=self.device)  # Exclude SOS/EOS



class Transliterator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder_embed = nn.Embedding(config.lat_vocab_size, config.embed_dim)
        self.encoder_rnn = getattr(nn, config.cell_type)(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.enc_layers,
            dropout=config.dropout if config.enc_layers > 1 else 0,
            batch_first=True
        )
        
        # Decoder
        self.decoder_embed = nn.Embedding(config.dev_vocab_size, config.embed_dim)
        self.decoder_rnn = getattr(nn, config.cell_type)(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.dec_layers,
            dropout=config.dropout if config.dec_layers > 1 else 0,
            batch_first=True
        )
        
        # Hidden state adapter
        # self.hidden_adapter = nn.Linear(
        #     config.hidden_dim * config.enc_layers,
        #     config.hidden_dim * config.dec_layers
        # )
        
        self.fc = nn.Linear(config.hidden_dim, config.dev_vocab_size)

    def _adapt_hidden(self, hidden):
        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            h = h[-self.config.dec_layers :]
            c = c[-self.config.dec_layers :]
            return (h.contiguous(), c.contiguous())
        else:  # GRU/RNN
            return hidden[-self.config.dec_layers :].contiguous()

    def _adapt_layer_dims(self, state):
        """Reshape hidden state to match decoder layers"""
        batch_size = state.size(1)
        
        # Flatten and project
        state = state.permute(1, 0, 2).contiguous()
        state = state.view(batch_size, -1)  # Flatten layers
        adapted = self.hidden_adapter(state)
        
        # Reshape to (num_layers, batch_size, hidden_dim)
        adapted = adapted.view(
            batch_size, 
            self.config.dec_layers, 
            self.config.hidden_dim
        ).permute(1, 0, 2)
        
        return adapted

    def encode(self, dev_input):
        embedded = self.encoder_embed(dev_input)
        _, hidden = self.encoder_rnn(embedded)
        return self._adapt_hidden(hidden)

    def decode_step(self, dev_input, hidden):
        embedded = self.decoder_embed(dev_input)
        output, hidden = self.decoder_rnn(embedded, hidden)
        logits = self.fc(output.squeeze(1))
        return logits, hidden

    def forward(self, lat_input, target_seq=None, beam_size=1):
        # Encoder
        embedded = self.encoder_embed(lat_input)
        _, hidden = self.encoder_rnn(embedded)
        
        # Adapt hidden state dimensions
        hidden = self._adapt_hidden(hidden)
        
        # Decoder
        if target_seq is not None:
            return self._teacher_force(hidden, target_seq)
        else:
            return self._beam_search(hidden, lat_input.size(0), beam_size)


    def _teacher_force(self, hidden, target_seq):
        batch_size = target_seq.size(0)
        seq_len = target_seq.size(1)
        
        outputs = torch.zeros(
            batch_size,
            seq_len,
            self.config.dev_vocab_size
        ).to(self.config.device)
    
        dec_input = target_seq[:, 0]
        for t in range(seq_len):
            dec_emb = self.decoder_embed(dec_input.unsqueeze(1))
            output, hidden = self.decoder_rnn(dec_emb, hidden)
            output = self.fc(output.squeeze(1))
            outputs[:, t] = output
            
            # Teacher forcing with boundary check
            if t+1 < target_seq.size(1):
                dec_input = target_seq[:, t+1] if torch.rand(1).item() < 0.5 else output.argmax(-1)
            else:
                dec_input = output.argmax(-1)

        return outputs


    def _beam_search(self, hidden, batch_size, beam_size):
        beams = [Beam(beam_size, self.config.device, self.config.eos_token)
                for _ in range(batch_size)]
        
        # Handle LSTM hidden states
        if isinstance(hidden, tuple):
            h, c = hidden
            hidden = (h.repeat(1, beam_size, 1), 
                    c.repeat(1, beam_size, 1))
        else:
            hidden = hidden.repeat(1, beam_size, 1)
        
        inputs = torch.full((batch_size * beam_size,), 1, 
                          device=self.config.device)
        
        for step in range(50):
            logits, hidden = self.decode_step(inputs.unsqueeze(1), hidden)
            log_probs = F.log_softmax(logits, dim=-1)
            
            new_inputs = []
            new_hidden = []
            all_done = True
            
            # Process each beam group
            for i in range(batch_size):
                beam = beams[i]
                beam_probs = log_probs[i*beam_size:(i+1)*beam_size]
                
                # Handle hidden state slicing
                if isinstance(hidden, tuple):
                    beam_h = (hidden[0][:, i*beam_size:(i+1)*beam_size, :],
                            hidden[1][:, i*beam_size:(i+1)*beam_size, :])
                else:
                    beam_h = hidden[:, i*beam_size:(i+1)*beam_size, :]
                
                done = beam.update(beam_probs, beam_h)
                if not done:
                    all_done = False
                    
                # Collect new hidden states
                beam_hidden = beam.get_current_hidden()
                if isinstance(beam_hidden, tuple):
                    new_hidden.append(beam_hidden)
                else:
                    new_hidden.append((beam_hidden,))
                
                # Collect new inputs
                new_inputs.extend(beam.get_current_tokens())
            
            # Update global hidden state
            if isinstance(hidden, tuple):
                # Concatenate LSTM states
                h_all = torch.cat([h[0] for h in new_hidden], dim=1)
                c_all = torch.cat([h[1] for h in new_hidden], dim=1)
                hidden = (h_all, c_all)
            else:
                # Concatenate GRU/RNN states
                hidden = torch.cat([h[0] for h in new_hidden], dim=1)
                
            inputs = torch.tensor(new_inputs, device=self.config.device)
            
            if all_done:
                break
        
        # return torch.stack([b.get_best_sequence() for b in beams])

        best_sequences = [b.get_best_sequence() for b in beams]
        
        # Pad sequences to match lengths
        max_len = max(seq.size(0) for seq in best_sequences)
        padded_sequences = []
        
        for seq in best_sequences:
            pad_length = max_len - seq.size(0)
            if pad_length > 0:
                padded_seq = torch.cat([
                    seq,
                    torch.zeros(pad_length, dtype=torch.long, device=seq.device)
                ])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        return torch.stack(padded_sequences)
        