from torch import nn
import torch

from src.base import BaseModel
from typing import List


def get_conv_shape(I, K, P, S, D=1):
    # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    numer = torch.tensor(I + 2 * P - D * (K - 1) - 1, dtype=torch.float64)
    return torch.floor(numer / S + 1)


class RNNBlock(nn.Module):
    def __init__(self, n_feats: int, hidden_size: int, use_batch_norm: bool, rnn_type: str):
        super().__init__()
        self.rnn = getattr(nn, rnn_type)(n_feats, hidden_size, batch_first=True, bias=False, bidirectional=True)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_size) # <-- takes (B, F, T)

    def forward(self, input: torch.Tensor):
        # (B, F, T)
        input, h = self.rnn(input.transpose(1, 2).contiguous())
        # (B, T, F * 2)
        input = input.view(input.shape[:-1] + (2, -1)).sum(-2).transpose(1, 2).contiguous()
        # (B, F, T)
        if self.use_batch_norm:
            input = self.batch_norm(input)
        # (B, F, T)
        return input


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, hidden_size, use_batch_norm: bool, rnn_type: str, num_rnn_layers: int, large: bool, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.TANH_MAX = 20
        self.large = large

        self.convs = self.get_conv_sequence(n_feats)
        self.rnns = self.get_rnn_sequence(self.conv_out_n_feats, hidden_size, use_batch_norm, rnn_type, num_rnn_layers)
        self.dense = nn.Linear(hidden_size, n_class, False)
    
    def conv_block(self, c_in, c_out, ker, stride, pad) -> List[nn.Module]:
        return [
            nn.Conv2d(c_in, c_out, ker, stride, pad),
            nn.BatchNorm2d(c_out),
            nn.Hardtanh(min_val=0, max_val=self.TANH_MAX, inplace=True)
            ]
    
    def get_conv_sequence(self, n_feats: int) -> nn.Sequential:
        # from DeepSpeech2 paper https://arxiv.org/pdf/1512.02595.pdf
        layers = self.conv_block(1, 32, (41, 11), (2, 2), (0, 0)) + \
                self.conv_block(32, 32, (21, 11), (2, 1), (0, 0))
        if self.large:
            layers += self.conv_block(32, 96, (21, 11), (2, 1), (0, 0))

        # calculating conv_out_n_feats
        for l in layers:
            if isinstance(l, nn.Conv2d):
                n_feats = get_conv_shape(n_feats,
                                         l.kernel_size[0],
                                         l.padding[0],
                                         l.stride[0],
                                         l.dilation[0])
                final_channel_num = l.out_channels

        self.conv_out_n_feats = int(n_feats * final_channel_num)
        return nn.Sequential(*layers)

    def get_rnn_sequence(self, n_feats: int, hidden_size: int, use_batch_norm: bool, rnn_type: str, num_rnn_layers: int) -> nn.Sequential:
        rnns = [RNNBlock(hidden_size if i > 0 else n_feats, hidden_size, use_batch_norm, rnn_type) for i in range(num_rnn_layers)]
        return nn.Sequential(*rnns)

    def forward(self, spectrogram, **batch):
        # (B, F, T)
        out = self.convs(spectrogram.unsqueeze(1))
        # (B, C, F_new, T)
        B, C, F_new, T = out.shape
        out = self.rnns(out.view(B, C * F_new, T)).transpose(1, 2).contiguous()
        # (B, T, F)
        out = self.dense(out)
        # (B, T, F)
        return {"logits": out}

    def transform_input_lengths(self, input_lengths: torch.Tensor):
        for l in self.convs:
            if isinstance(l, nn.Conv2d):
                input_lengths = get_conv_shape(input_lengths,
                                                l.kernel_size[1],
                                                l.padding[1],
                                                l.stride[1],
                                                l.dilation[1]).type(torch.int32)
        return input_lengths
