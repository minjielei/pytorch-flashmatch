import numpy as np
import torch
import yaml
from toymc import ToyMC
from algorithms.match_model import GradientModel, PoissonMatchLoss, EarlyStopping
from utils import get_x_constraints

class FlashMatchManager():
    """
    Top level FlashMatchManager program that runs the io and matching algorithm
    """
    def __init__(self, detector_cfg, photon_library, cfg):
        self.configure(detector_cfg, photon_library, cfg)

    def configure(self, photon_library, detector_file, cfg):
        config = yaml.load(open(cfg), Loader=yaml.Loader)['FlashMatchManager']
        self.detector_specs = yaml.load(open(detector_file), Loader=yaml.Loader)['DetectorSpecs']
        self.max_iteration = int(config['MaxIteration'])
        self.init_lr = config['InitLearningRate']
        self.min_lr = config['MinLearningRate']
        self.patience = config['Patience']
        self.min_delta = config['MinDelta']
        self.loss_threshold = config['LossThreshold']
        self.lr_multiplier = config['LRateMultipliter']

        self.reader = ToyMC(photon_library, detector_file, cfg)
        self.flash_algo = self.reader.flash_algo
        self.loss_fn = PoissonMatchLoss()
        self.early_stopping = EarlyStopping(self.patience, self.min_delta)
        self.model = None
        self.optimizer = None
        

    def make_flashmatch_input(self, num_tracks):
        return self.reader.make_flashmatch_input(num_tracks)

    def train(self, input, target):
        constraints = get_x_constraints(input, self.detector_specs) 
        self.model = GradientModel(self.flash_algo, constraints)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, min_lr=self.min_lr)

        for i in range(self.max_iteration):
            pred = self.model(input)
            loss, match = self.loss_fn(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step(loss)
            # self.early_stopping(loss)
            # if self.early_stopping.early_stop:
            #     # if loss > self.loss_threshold:
            #     #     optimizer.param_groups[0]['lr'] *= self.lr_multiplier
            #     #     self.early_stopping = EarlyStopping(self.patience, self.min_delta)
            #     # else:
            #     #     break
            #     break

        return loss.item(), match.item(), self.model.xshift.x.item()

    def run_flash_match(self, flashmatch_input):
        pass