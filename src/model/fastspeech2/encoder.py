from torch import nn
from torch import Tensor
import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
from src.model.fastspeech2.ff_transformer import FFTBlock


def get_non_pad_mask(seq, PAD):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, PAD):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


class Encoder(nn.Module):
    def __init__(self, encoder_dim, encoder_head, encoder_n_layer, encoder_conv1d_filter_size,
                 fft_conv1d_kernel, fft_conv1d_padding, vocab_size, max_seq_len, PAD, dropout):
        super(Encoder, self).__init__()
        
        len_max_seq=max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer
        self.PAD = PAD

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_head,
            encoder_dim // encoder_head,
            encoder_dim // encoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(src_seq, PAD=self.PAD)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        return enc_output, non_pad_mask
