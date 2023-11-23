from torch import Tensor, zeros_like
from torch.nn import CrossEntropyLoss
from torch.nn.functional import pad
from torch.nn import Module
import torch


def zero_mean(t: torch.Tensor):
    return t - torch.mean(t, dim=-1, keepdim=True)

def si_sdr(est: Tensor, target: Tensor, **kwargs):
    alpha = (torch.sum(target * est, dim=-1) / torch.square(torch.linalg.norm(target, dim=-1))).unsqueeze(1)
    return 20 * torch.log10(torch.linalg.norm(alpha * target, dim=-1) / (torch.linalg.norm(alpha * target - est, dim=-1) + 1e-6) + 1e-6)

# def to_real_length(t: Tensor, mixed_lens: Tensor) -> Tensor:
#     masked = zeros_like(t)
#     for row, len in enumerate(mixed_lens):
#         masked[row, :len] = t[row, :len]
#     return masked

class SpexLoss(Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = CrossEntropyLoss()

    def forward(self, is_train: bool, **batch) -> Tensor:
        num_samples = batch["target"].shape[0]

        # target = to_real_length(batch["target"], batch["mixed_lens"])
        # pred_short = to_real_length(batch["pred_short"], batch["mixed_lens"])
        # pred_mid = to_real_length(batch["pred_mid"], batch["mixed_lens"])
        # pred_long = to_real_length(batch["pred_long"], batch["mixed_lens"])

        short_si_sdr = si_sdr(batch["pred_short"], batch["target"]).sum()
        mid_si_sdr = si_sdr(batch["pred_mid"], batch["target"]).sum()
        long_si_sdr = si_sdr(batch["pred_long"], batch["target"]).sum()

        si_sdr_loss = -((1 - self.alpha - self.beta) * short_si_sdr + self.alpha * mid_si_sdr + self.beta * long_si_sdr) / num_samples
        if not is_train:
            return si_sdr_loss

        ce_loss = self.ce(batch["speakers_logits"], batch["speakers_ids"])
        return si_sdr_loss + self.gamma * ce_loss
