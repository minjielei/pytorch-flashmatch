import torch
from .match_layers import GenFlash, XShift

class GradientModel(torch.nn.Module):
    def __init__(self, flash_algo, cfg=None):
        super(GradientModel, self).__init__()
        self.model_cfg = cfg
        self.flash_algo = flash_algo
        self.xshift = XShift()
        self.genflash = GenFlash.apply

    def forward(self, input):
        x = self.xshift(input)
        flash = self.genflash(x, self.flash_algo)
        return flash