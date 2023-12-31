from torch import nn
from torch import Tensor
import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class Predictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(Predictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = duration_predictor_filter_size
        self.kernel = duration_predictor_kernel_size
        self.conv_output_size = duration_predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count + k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    """ Length Regulator """
    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = Predictor(encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                    duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        """alpha: float, controls the length of the expanded sequence (audio speed)
        """
        log_duration = self.duration_predictor(x)

        if target is None:
            duration = torch.exp(log_duration) - 1
            # to round up adding 0.5
            assert torch.all(duration >= 0), "duration cannot be less 0"
            duration = ((alpha * duration) + 0.5).int()
            output = self.LR(x, duration)
            spec_pos = torch.stack(
                [torch.Tensor([i for i in range(1, output.shape[1] + 1)]).to(x.device)]
                ).long()
            return output, spec_pos
        else:
            # if we have target we return log_dur for loss calculation
            output = self.LR(x, target, mel_max_length)
            return output, log_duration
