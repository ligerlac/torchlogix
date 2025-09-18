import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableThermometerThresholding(nn.Module):
    def __init__(self, init_thresholds, slope=10.0):
        super().__init__()
        self.num_thresholds = len(init_thresholds)
        self.slope = slope
        self._frozen = False  # switch to control hard/soft behavior

        init_t = torch.tensor(init_thresholds, dtype=torch.float32)
        first = init_t[:1]
        diffs = torch.diff(init_t, prepend=first.new_zeros(1))
        self.raw_diffs = nn.Parameter(diffs)

    def get_thresholds(self):
        if self._frozen:
            return torch.cumsum(self.raw_diffs, dim=0)
        else:
            diffs_pos = F.softplus(self.raw_diffs)
            return torch.cumsum(diffs_pos, dim=0)

    def freeze_thresholds(self):
        with torch.no_grad():
            thresholds = self.get_thresholds().round()
            first = thresholds[:1]
            diffs = torch.diff(thresholds, prepend=first.new_zeros(1))
            self.raw_diffs.copy_(diffs)
        self.raw_diffs.requires_grad = False
        self._frozen = True

    def forward(self, x):
        thresholds = self.get_thresholds()  # (T,)
        if x.ndim == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # -> (B, 1, H, W)
        thresholds = thresholds.view(1, -1, 1, 1)

        if self._frozen:
            # Hard thermometer encoding
            outputs = (x > thresholds).float()
        else:
            # Soft, differentiable approximation
            outputs = torch.tanh(self.slope * (x - thresholds))
            outputs = (outputs + 1.0) / 2.0
        return outputs