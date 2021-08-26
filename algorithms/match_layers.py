import torch
from torch.autograd import grad
import torch.nn as nn
from .siren_modules import Siren
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class XShift(nn.Module):
    """
        Shift input qcluster along x axis
    """
    def __init__(self, dx0, dx_min, dx_max):
        super(XShift, self).__init__()
        self.dx_min = dx_min
        self.dx_max = dx_max
        self.dx = nn.Parameter(torch.empty(1))
        self.dx.data.fill_(dx0)

    def forward(self, input):
        self.dx.data.clamp_(self.dx_min, self.dx_max)
        shift = torch.cat((self.dx, torch.zeros(3, device=device)), -1)
        return torch.add(input, shift.expand(input.shape[0], -1))


class GenFlash(torch.autograd.Function):
    """
        Custom autograd function to generate flash hypothesis
    """

    @staticmethod
    def forward(ctx, input, flash_algo):
        ctx.save_for_backward(input)
        ctx.flash_algo = flash_algo
        return flash_algo.fill_estimate(input)

    @staticmethod
    def backward(ctx, grad_output):
        track = ctx.saved_tensors[0]
        grad_plib = ctx.flash_algo.backward_gradient(track)
        grad_input = torch.matmul(grad_plib, grad_output.unsqueeze(-1))
        pad = torch.zeros(grad_input.shape[0], 3, device=device)
        return torch.cat((grad_input, pad), -1), None
        
class SirenFlash(nn.Module):
    def __init__(self, flash_algo, in_features=3, out_features=180, hidden_features=256, hidden_layers=1, outermost_linear=True, omega=5):
        super().__init__()
        self.flash_algo = flash_algo
        self.model = Siren(in_features, out_features, hidden_features, hidden_layers, outermost_linear, omega)
        self.model = self.model.float()
        self.model = torch.nn.DataParallel(self.model, device=device)
        self.model.cuda()
        self.model.load_state_dict(torch.load(flash_algo.siren_path))

    def forward(self, input):
        local_pe_v = torch.sum(self.plib.VisibilityFromXYZ(input[:, :3])*(input[:, 3].unsqueeze(-1)), axis = 0)
        if len(self.qe_v) == 0:
          self.qe_v = torch.ones(local_pe_v.shape, device=device)
        return local_pe_v * self.flash_algo.global_qe * self.flash_algo.reco_pe_calib / self.flash_algo.qe_v
