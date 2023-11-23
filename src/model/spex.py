from torch import nn
import torch
import torch.nn.functional as F

from src.base import BaseModel
from typing import List, Tuple


def get_conv_shape(I, K, P, S, D=1):
    # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    numer = torch.tensor(I + 2 * P - D * (K - 1) - 1, dtype=torch.float64)
    return torch.floor(numer / S + 1)

def get_len_after_conv(T: int, L_init: int, L: int):
    # K = 2 * (T - L) / L + 1
    return L + (L_init // 2) * (T - 1)

class GlobalLayerNorm(nn.Module):
    eps: float = 1e-05

    def __init__(self, normalized_shape: Tuple[int, ...]):
        super(GlobalLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        stat_axes = (1, 2)
        mean = torch.mean(x, stat_axes, keepdim=True)
        std = torch.sqrt(torch.var(x, stat_axes, keepdim=True, unbiased=True) + self.eps)
        return (x - mean) / std * self.weight + self.bias


class TCNBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, speaker_emb_size: int, de_cnn_k: int, de_cnn_dil: int, first_in_stack: bool, **kwargs):
        super(TCNBlock, self).__init__()

        self.first_in_stack = first_in_stack
        if first_in_stack:
            assert speaker_emb_size is not None, "speaker_emb_size should not be None if first_in_stack"
            input_channels = speaker_emb_size + in_channels
        else:
            assert speaker_emb_size is None, "speaker_emb_size should be None if not first_in_stack"
            input_channels = in_channels
        
        self.conv1x1_in = nn.Conv1d(input_channels, hidden_channels, 1)
        self.conv1x1_out = nn.Conv1d(hidden_channels, in_channels, 1)

        self.act_norm1 = nn.Sequential(nn.PReLU(), GlobalLayerNorm((hidden_channels, 1)))
        self.act_norm2 = nn.Sequential(nn.PReLU(), GlobalLayerNorm((hidden_channels, 1)))

        # depthwise convolution
        self.de_cnn = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            groups=hidden_channels,
            padding=(de_cnn_k - 1) * de_cnn_dil // 2,
            kernel_size=de_cnn_k,
            dilation=de_cnn_dil
        )
    
    def forward(self, tuple: Tuple):
        x, speaker_emb = tuple

        if self.first_in_stack:
            # concat x with speakers embedding
            out = torch.cat((x, speaker_emb.unsqueeze(-1).repeat(1, 1, x.shape[-1])), dim=1)
        else:
            out = x
        
        out = self.conv1x1_in(out)
        out = self.act_norm1(out)
        out = self.de_cnn(out)
        out = self.act_norm2(out)
        out = self.conv1x1_out(out)
        return out + x, speaker_emb


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1x1_before = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv1x1_after = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.act_norm = nn.Sequential(nn.PReLU(), nn.BatchNorm1d(out_channels))
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.act_norm(self.conv1x1_before(x))
        out = self.bn(self.conv1x1_after(out))
        if hasattr(self, "residual_conv"):
            x = self.residual_conv(x)
        out = self.prelu(out + x)
        out = self.maxpool(out)
        return out
    

class SpeakerEncoder(nn.Module):
    def __init__(self, in_channels, representation_channels, de_cnn_channels, speaker_emb_size):
        super(SpeakerEncoder, self).__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.conv1x1_in = nn.Conv1d(in_channels, representation_channels, 1)
        self.conv1x1_out = nn.Conv1d(de_cnn_channels, speaker_emb_size, 1)

        self.resnet = nn.Sequential(
            ResNetBlock(representation_channels, representation_channels),
            ResNetBlock(representation_channels, de_cnn_channels),
            ResNetBlock(de_cnn_channels, de_cnn_channels)
        )
    
    def forward(self, x: torch.Tensor):
        assert x.dim() == 3, f"x dim {x.dim()} should be 3"
        # x: BxCxT
        # transpose to do layer normalization along channels
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv1x1_in(x)
        x = self.resnet(x)
        x = self.conv1x1_out(x)
        return x


class SpexPlus(nn.Module):
    def __init__(self, L1, L2, L3, speaker_emb_size, num_classes, num_filters, num_blocks, num_stacks, representation_channels, de_cnn_channels, de_cnn_k, **batch):
        super().__init__(**batch)

        # N = num_filters
        # B = num_blocks
        # O = representation_channels
        # P = de_cnn_channels
        # Q = de_cnn_k
        self.representation_channels = representation_channels
        self.speaker_emb_size = speaker_emb_size
        self.num_blocks = num_blocks
        self.de_cnn_channels = de_cnn_channels
        self.de_cnn_k = de_cnn_k
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        # speech encoder
        enc_stride = L1 // 2
        self.enc_short = nn.Sequential(nn.ReLU(), nn.Conv1d(1, num_filters, L1, enc_stride))
        self.enc_mid = nn.Sequential(nn.ReLU(), nn.Conv1d(1, num_filters, L2, enc_stride))
        self.enc_long = nn.Sequential(nn.ReLU(), nn.Conv1d(1, num_filters, L3, enc_stride))

        self.enc_norm = nn.LayerNorm(3 * num_filters)
        self.enc_conv1x1 = nn.Conv1d(3 * num_filters, representation_channels, 1)

        # speaker encoder
        self.speaker_enc = SpeakerEncoder(3 * num_filters, representation_channels, de_cnn_channels, speaker_emb_size)
        self.classifier = nn.Linear(speaker_emb_size, num_classes)

        # speaker extractor
        self.tcn_blocks = nn.Sequential(*[
            self.tcn_stack() for _ in range(num_stacks)
        ])

        self.mask_short = nn.Sequential(nn.ReLU(), nn.Conv1d(representation_channels, num_filters, 1))
        self.mask_mid = nn.Sequential(nn.ReLU(), nn.Conv1d(representation_channels, num_filters, 1))
        self.mask_long = nn.Sequential(nn.ReLU(), nn.Conv1d(representation_channels, num_filters, 1))

        # speech decoder
        self.dec_short = nn.ConvTranspose1d(num_filters, 1, L1, enc_stride)
        self.dec_mid = nn.ConvTranspose1d(num_filters, 1, L2, enc_stride)
        self.dec_long = nn.ConvTranspose1d(num_filters, 1, L3, enc_stride)
    
    def tcn_stack(self):
        stack = [
            TCNBlock(
                in_channels=self.representation_channels,
                hidden_channels=self.de_cnn_channels,
                first_in_stack=(i == 0),
                de_cnn_dil=(2 ** i),
                de_cnn_k=self.de_cnn_k,
                speaker_emb_size=self.speaker_emb_size if i == 0 else None,

            ) for i in range(self.num_blocks)
        ]
        return nn.Sequential(*stack)

    def get_real_ref_lens(self, ref_lens: torch.Tensor):
        ref_lens = (1 + (ref_lens - self.L1) // (self.L1 // 2)).view(-1, 1)
        ref_lens = ((ref_lens // 3) // 3) // 3
        return ref_lens.float()

    @staticmethod
    def avg_pool(x: torch.Tensor, lens: torch.Tensor):
        return x.sum(2) / lens

    def forward(self, mixed: torch.Tensor, mixed_lens: torch.Tensor, ref: torch.Tensor, ref_lens: torch.Tensor, **batch):
        # ref: BxL
        mixed = mixed.unsqueeze(1)
        # mixed: Bx1xL
        assert mixed.dim() == 3, f"mixed dim {mixed.dim()} should be 3"

        # speech encoder
        mixed_short = self.enc_short(mixed)
        mid_enc_length = get_len_after_conv(mixed_short.shape[2], self.L1, self.L2)
        long_enc_length = get_len_after_conv(mixed_short.shape[2], self.L1, self.L3)
        mixed_mid = self.enc_mid(F.pad(mixed, (0, mid_enc_length - mixed.shape[2])))
        mixed_long = self.enc_long(F.pad(mixed, (0, long_enc_length - mixed.shape[2])))
        mixed_enc = self.enc_conv1x1(self.enc_norm(
            torch.cat((mixed_short, mixed_mid, mixed_long), dim=1).transpose(1, 2)).transpose(1, 2))

        # speaker encoder
        ref_short = self.enc_short(ref.unsqueeze(1))
        mid_enc_length = get_len_after_conv(ref_short.shape[2], self.L1, self.L2)
        long_enc_length = get_len_after_conv(ref_short.shape[2], self.L1, self.L3)
        ref_mid = self.enc_mid(F.pad(mixed, (0, mid_enc_length - mixed.shape[2])))
        ref_long = self.enc_long(F.pad(mixed, (0, long_enc_length - mixed.shape[2])))
        ref_enc = self.speaker_enc(torch.cat((ref_short, ref_mid, ref_long), dim=1))
        # ref_enc: B x emb_size x L
        ref_lens = self.get_real_ref_lens(ref_lens)
        ref_enc = SpexPlus.avg_pool(ref_enc, ref_lens) # B x emb_size x 1
        ref_logits = self.classifier(ref_enc) # B x num_classes

        # speaker extractor
        mixed_enc = self.tcn_blocks((mixed_enc, ref_enc))[0] # (x, spk_emb)[0]
        pred_short = mixed_short * self.mask_short(mixed_enc)
        pred_mid = mixed_mid * self.mask_mid(mixed_enc)
        pred_long = mixed_long * self.mask_long(mixed_enc)

        # speech decoder
        pred_short = self.dec_short(pred_short).squeeze(1)
        pred_short = F.pad(pred_short, (0, mixed.shape[-1] - pred_short.shape[-1]))
        pred_mid = self.dec_mid(pred_mid).squeeze(1)[..., :mixed.shape[-1]]
        pred_long = self.dec_long(pred_long).squeeze(1)[..., :mixed.shape[-1]]

        return {"pred_short": pred_short,
                "pred_mid": pred_mid,
                "pred_long": pred_long,
                "speakers_logits": ref_logits}
