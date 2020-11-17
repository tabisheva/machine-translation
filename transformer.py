import torch
import torch.nn as nn
import math


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size, device, d_model=256,
                 nhead=8, num_enc_layers=4, num_dec_layers=4,
                 dim_feedforward=512, dropout=0.1, activation='relu', pad_index=1):
        super(Transformer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.pad_index = pad_index
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PostionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_enc_layers,
                                          num_dec_layers, dim_feedforward,
                                          dropout, activation)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def get_pad_mask(self, data):
        mask = data.eq(self.pad_index).transpose(0, 1)
        mask = mask.masked_fill(mask == True, float(
            '-inf')).masked_fill(mask == False, float(0.0))
        return mask

# https://github.com/pytorch/tutorials/blob/e5f60c64367b2ff3c19e5b73a2c7eac4bf1269f1/beginner_source/transformer_tutorial.py#L67
    def get_square_subsequent_mask(self, tgt):
        seq_len = tgt.size(0)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.float().masked_fill(mask == 0, float(
            0.0)).masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src, tgt):
        src, _, src_pad_mask = self.get_masks(src)
        tgt, tgt_subsequent_mask, tgt_pad_mask = self.get_masks(tgt)
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_subsequent_mask,
                               src_key_padding_mask=src_pad_mask,
                               tgt_key_padding_mask=tgt_pad_mask,
                               memory_key_padding_mask=src_pad_mask)
        out = out.transpose(0, 1)
        out = self.out(out)
        return out

    def get_masks(self, x):
        x = x.transpose(0, 1)
        x_subsequent_mask = self.get_square_subsequent_mask(
            x).to(self.device)
        x_pad_mask = self.get_pad_mask(x).to(self.device)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoder(x))
        return x, x_subsequent_mask, x_pad_mask

# https://github.com/pytorch/tutorials/blob/e5f60c64367b2ff3c19e5b73a2c7eac4bf1269f1/beginner_source/transformer_tutorial.py#L94
class PostionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PostionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(- torch.arange(0, d_model,
                                            2).float() * math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
