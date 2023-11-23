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


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, decoder_dim, decoder_head, decoder_n_layer, decoder_conv1d_filter_size,
                 fft_conv1d_kernel, fft_conv1d_padding, max_seq_len, PAD, dropout):
        super(Decoder, self).__init__()

        len_max_seq=max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer
        self.PAD = PAD

        self.position_enc = nn.Embedding(
            n_position,
            decoder_dim,
            padding_idx=PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_head,
            decoder_dim // decoder_head,
            decoder_dim // decoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):
        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, PAD=self.PAD)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
