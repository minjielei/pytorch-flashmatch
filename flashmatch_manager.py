import numpy as np
import torch
from torch._C import device
import yaml
from toymc import ToyMC
from algorithms.match_model import GradientModel, PoissonMatchLoss, EarlyStopping
from utils import get_x_constraints
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.lr_range = config['LearningRateRange']
        self.loss_thresholds = config['LossThresholds']
        self.min_lr = config['MinLearningRate']
        self.scheduler_factor = config['SchedulerFactor']
        self.stopping_patience = config['StoppingPatience']
        self.stopping_delta = config['StoppingDelta']

        self.reader = ToyMC(photon_library, detector_file, cfg)
        self.flash_algo = self.reader.flash_algo
        self.loss_fn = PoissonMatchLoss()

    def make_flashmatch_input(self, num_tracks):
        return self.reader.make_flashmatch_input(num_tracks)

    def train(self, input, target, true_x):
        constraints = get_x_constraints(input, self.detector_specs) 
        self.model = GradientModel(self.flash_algo, constraints)
        self.model.to(device)
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter()

        # run one iteration to determine optimal initial learning rate
        pred = self.model(input)
        loss, match = self.loss_fn(pred, target)
        for lr, loss_threshold in zip(self.lr_range, self.loss_thresholds):
            if loss < loss_threshold:
                self.init_lr = lr
                break

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, min_lr=self.min_lr, factor=self.scheduler_factor)
        early_stopping = EarlyStopping(self.stopping_patience, self.stopping_delta)

        for i in range(self.max_iteration):
            pred = self.model(input)
            loss, match = self.loss_fn(pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            scheduler.step(loss)
            early_stopping(loss)

            # writer.add_scalar("Loss", loss, i)
            # writer.add_scalar("Reco X - True X", self.model.xshift.x - true_x, i)
            # writer.add_scalar("Reco PE - True PE", torch.sum(pred) - torch.sum(target[match.item()]), i)
            if early_stopping.early_stop:
                break

        return loss.item(), match.item(), self.model.xshift.x.item()

    def run_flash_match(self, flashmatch_input):
        pass