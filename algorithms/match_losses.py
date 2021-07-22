import torch
import torch.nn as nn

class PoissonMatchLoss(nn.Module):
    def __init__(self):
        super(PoissonMatchLoss, self).__init__()
        self.poisson_nll = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')

    def forward(self, input, target):
        loss, match = torch.min(torch.sum(self.poisson_nll(input.expand(target.shape[0], -1), target), axis=1), 0)
        return loss, match

