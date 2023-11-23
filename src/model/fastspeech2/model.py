from torch import nn
import torch
import torch.nn.functional as F
from torch import bucketize
import numpy as np

from typing import List, Tuple
from src.model.fastspeech2.encoder import Encoder
from src.model.fastspeech2.decoder import Decoder
from src.model.fastspeech2.predictors import LengthRegulator, Predictor


def get_conv_shape(I, K, P, S, D=1):
    # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    numer = torch.tensor(I + 2 * P - D * (K - 1) - 1, dtype=torch.float64)
    return torch.floor(numer / S + 1)

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2(nn.Module):
    def __init__(self,
                # fft:
                fft_conv1d_kernel: Tuple,
                fft_conv1d_padding: Tuple,
                # encoder:
                encoder_dim,
                encoder_head,
                encoder_n_layer,
                encoder_conv1d_filter_size,
                # decoder:
                decoder_dim,
                decoder_head,
                decoder_n_layer,
                decoder_conv1d_filter_size,
                # variance adaptor:
                pitch_filter_size, pitch_kernel_size,
                energy_filter_size, energy_kernel_size,
                dur_filter_size, dur_kernel_size,
                # variance space:
                n_bins, n_mels,
                pitch_min, pitch_max,
                energy_min, energy_max,
                # sequence:
                max_seq_len,
                vocab_size,
                PAD,
                dropout):
        super().__init__()

        # feed-forward transformer
        self.encoder = Encoder(encoder_dim, encoder_head, encoder_n_layer, encoder_conv1d_filter_size, fft_conv1d_kernel, fft_conv1d_padding, vocab_size, max_seq_len, PAD, dropout)
        self.decoder = Decoder(decoder_dim, decoder_head, decoder_n_layer, decoder_conv1d_filter_size, fft_conv1d_kernel, fft_conv1d_padding, max_seq_len, PAD, dropout)

        # variance adaptor
        self.length_regulator = LengthRegulator(encoder_dim, dur_filter_size, dur_kernel_size, dropout)

        # adding 1 to make log more stable
        self.register_buffer("pitch_bins", torch.linspace(np.log1p(pitch_min), np.log1p(pitch_max), n_bins - 1))
        self.pitch_embed = nn.Embedding(n_bins, encoder_dim)

        self.register_buffer("energy_bins", torch.linspace(np.log1p(energy_min), np.log1p(energy_max), n_bins - 1))
        self.energy_embed = nn.Embedding(n_bins, encoder_dim)

        self.pitch_predictor = Predictor(encoder_dim, pitch_filter_size, pitch_kernel_size, dropout)
        self.energy_predictor = Predictor(encoder_dim, energy_filter_size, energy_kernel_size, dropout)
        
        self.mel_linear = nn.Linear(decoder_dim, n_mels)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
    
    def get_energy_embedding(self, x, gamma, energy_target=None):
        energy = self.energy_predictor(x)
        if energy_target is not None:
            buckets = bucketize(torch.log1p(energy_target), self.energy_bins)
        else:
            energy = torch.log1p(torch.expm1(energy) * gamma)
            buckets = bucketize(energy, self.energy_bins)
        
        return energy, self.energy_embed(buckets)
    
    def get_pitch_embedding(self, x, beta, pitch_target=None):
        pitch = self.pitch_predictor(x)
        if pitch_target is not None:
            buckets = bucketize(torch.log1p(pitch_target), self.pitch_bins)
        else:
            pitch = torch.log1p(torch.expm1(pitch) * beta)
            buckets = bucketize(pitch, self.pitch_bins)
        
        return pitch, self.pitch_embed(buckets)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, pitch_target=None,
                energy_target=None, dur_target=None, alpha=1.0, beta=1.0, gamma=1.0, **batch):
        if self.training:
            enc_output = self.encoder(src_seq, src_pos)[0]
            dur_emb, log_dur = self.length_regulator(enc_output, alpha, dur_target, mel_max_length)
            energy, energy_emb = self.get_energy_embedding(dur_emb, gamma, energy_target)
            log_pitch, pitch_emb = self.get_pitch_embedding(dur_emb, beta, pitch_target)

            variance_emb = dur_emb + energy_emb + pitch_emb
            dec_output = self.decoder(variance_emb, mel_pos)
            mel_pred = self.mel_linear(self.mask_tensor(dec_output, mel_pos, mel_max_length))
            return {"mel_pred": mel_pred, "log_dur_pred": log_dur, "energy_pred": energy, "log_pitch_pred": log_pitch}
        else:
            enc_output = self.encoder(src_seq, src_pos)[0]
            dur_emb, mel_pos = self.length_regulator(enc_output, alpha)
            energy_emb = self.get_energy_embedding(dur_emb, gamma)[1]
            pitch_emb = self.get_pitch_embedding(dur_emb, beta)[1]
            
            variance_emb = dur_emb + energy_emb + pitch_emb
            mel_pred = self.mel_linear(self.decoder(variance_emb, mel_pos))
            return {"mel_pred": mel_pred}
