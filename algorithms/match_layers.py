import torch
from torch.autograd import grad
import torch.nn as nn

class XShift(nn.Module):
    """
        Shift input qcluster along x axis
    """
    def __init__(self, constraints):
        super(XShift, self).__init__()
        self.x_min = constraints[0]
        self.x_max = constraints[1]
        self.ic = (self.x_min + self.x_max) / 2.
        self.x = nn.Parameter(torch.empty(1))
        self.x.data.fill_(self.ic)

    def forward(self, input):
        self.x.data.clamp_(self.x_min, self.x_max)
        shift = torch.cat((self.x, torch.zeros(3)), -1)
        return torch.add(input, shift.expand(input.shape[0], -1))


class GenFlash(torch.autograd.Function):
    """
        Custom autograd function to generate flash hypothesis
    """

    @staticmethod
    def forward(ctx, input, flash_algo):
        ctx.save_for_backward(input)
        ctx.flash_algo = flash_algo
        track = input.detach().numpy()
        return torch.Tensor(flash_algo.fill_estimate(track, use_numpy=True))

    @staticmethod
    def backward(ctx, grad_output):
        track = ctx.saved_tensors[0].detach().numpy()
        grad_plib = torch.Tensor(ctx.flash_algo.backward_gradient(track))
        grad_input = torch.matmul(grad_plib, grad_output.unsqueeze(-1))
        pad = torch.zeros(grad_input.shape[0], 3)
        return torch.cat((grad_input, pad), -1), None
        

