import torch
import torch.nn as nn

class PoissonMatchLoss(nn.Module):
    def __init__(self):
        super(PoissonMatchLoss, self).__init__()
        self.poisson_nll = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')

    def forward(self, input, target):
        input_v = torch.transpose(input.repeat(target.shape[0], 1, 1), 0, 1)
        target_v = target.repeat(input.shape[0], 1, 1)
        loss_v, match = torch.min(torch.sum(self.poisson_nll(input_v, target_v), axis=2), 1)
        loss = torch.mean(loss_v)
        return loss, match

