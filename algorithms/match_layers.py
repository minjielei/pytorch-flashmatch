import torch
from torch.autograd import grad
import torch.nn as nn

class XShift(nn.Module):
    """
        Shift input qcluster along x axis
    """
    def __init__(self, num_tracks):
        super(XShift, self).__init__()
        self.num_tracks = num_tracks
        self.x = nn.Parameter(torch.empty(num_tracks, 1))
        self.x.data.fill_(0)

    def forward(self, input):
        shift = torch.cat((self.x, torch.zeros(self.num_tracks, input.shape[2]-1)), -1)
        return torch.add(input, shift.expand(input.shape[1], -1, -1).reshape(input.shape))


class GenFlash(torch.autograd.Function):
    """
        Custom autograd function to generate flash hypothesis
    """

    @staticmethod
    def forward(ctx, input, flash_algo):
        ctx.save_for_backward(input)
        ctx.flash_algo = flash_algo
        return torch.Tensor([flash_algo.fill_estimate(track, use_numpy=True) for track in input.detach().numpy()])

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0].detach().numpy()
        grad_plib = torch.Tensor([ctx.flash_algo.backward_gradient(track) for track in input])
        grad_input = torch.matmul(grad_plib, grad_output.unsqueeze(-1))
        pad = torch.zeros(grad_input.shape)
        return torch.cat((grad_input, pad, pad, pad), -1), None
        

