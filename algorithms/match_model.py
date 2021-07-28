import torch
import torch.nn as nn
from .match_layers import GenFlash, XShift

class GradientModel(torch.nn.Module):
    """
    Gradient-based optimization model for flash matching
    """
    def __init__(self, flash_algo, constraints):
        super(GradientModel, self).__init__()
        self.flash_algo = flash_algo
        self.xshift = XShift(constraints)
        self.genflash = GenFlash.apply

    def forward(self, input):
        x = self.xshift(input)
        flash = self.genflash(x, self.flash_algo)
        return flash

class PoissonMatchLoss(nn.Module):
    """
    Poisson NLL Loss for gradient-based optimization model
    """
    def __init__(self):
        super(PoissonMatchLoss, self).__init__()
        self.poisson_nll = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')

    def forward(self, input, target):
        loss, match = torch.min(torch.mean(self.poisson_nll(input.expand(target.shape[0], -1), target), axis=1), 0)
        return loss, match

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain iterations.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, loss):
        if self.best_loss == None:
            self.best_loss = loss
        elif self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True