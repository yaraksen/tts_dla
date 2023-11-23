import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss as mse


class FSLoss(Module):
    def forward(self, mel_pred, mel_target, log_dur_pred, energy_pred, log_pitch_pred, dur_target, energy_target, pitch_target,
                **batch) -> dict:
        return {
            "mse_loss": mse(mel_pred, mel_target),
            "dur_loss": mse(log_dur_pred, torch.log1p(dur_target.float())),
            "energy_loss": mse(energy_pred, torch.log1p(energy_target)),
            "pitch_loss": mse(log_pitch_pred, torch.log1p(pitch_target))
        }
