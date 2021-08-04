import numpy as np
import torch
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
        self.num_processes = config['NumProcesses']

        self.reader = ToyMC(photon_library, detector_file, cfg)
        self.flash_algo = self.reader.flash_algo
        self.loss_fn = PoissonMatchLoss()

    def make_flashmatch_input(self, num_tracks):
        return self.reader.make_flashmatch_input(num_tracks)

    def train(self, input, target=None):
        if target is None:
            target = self.target
        constraints = get_x_constraints(input, self.detector_specs) 
        model = GradientModel(self.flash_algo, constraints)
        model.to(device)
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter()

        # run one iteration to determine optimal initial learning rate
        pred = model(input)
        loss, match = self.loss_fn(pred, target)
        for lr, loss_threshold in zip(self.lr_range, self.loss_thresholds):
            if loss < loss_threshold:
                self.init_lr = lr
                break

        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=self.min_lr, factor=self.scheduler_factor)
        early_stopping = EarlyStopping(self.stopping_patience, self.stopping_delta)

        for i in range(self.max_iteration):
            pred = model(input)
            loss, match = self.loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            early_stopping(loss)

            # writer.add_scalar("Loss", loss, i)
            # writer.add_scalar("Reco X - True X", self.model.xshift.x - true_x, i)
            # writer.add_scalar("Reco PE - True PE", torch.sum(pred) - torch.sum(target[match.item()]), i)
            if early_stopping.early_stop:
                break

        return loss.item(), match.item(), model.xshift.x.item(), torch.sum(pred).item()

    def match(self, flashmatch_input):
        
        track_v, flash_v = flashmatch_input.make_torch_input()
        self.target = flash_v
        matches = []
        reco_x = []
        idx = 0

        import torch.multiprocessing as mp
        from multiprocessing.pool import ThreadPool
        ctx = mp.get_context("spawn")
        with ThreadPool(processes=self.num_processes) as pool:
            for loss, match, x, pe in pool.imap(self.train, track_v):
                matches.append(match)
                reco_x.append(x)

                track_id = flashmatch_input.qcluster_v[idx].idx
                true_x = flashmatch_input.x_shift[idx]
                true_pe = np.sum(flashmatch_input.flash_v[track_id])
                self.print_match_result(idx, track_id, match, loss, true_x, x, true_pe, pe)
                idx += 1
            
        return matches, reco_x

    def print_match_result(self, id, track_id, flash_id, loss, true_x, reco_x, true_pe, reco_pe):
        print('Match ID: ', id)
        correct = (track_id == flash_id)
        template = """TPC/PMT IDs {}/{} Correct? {}, Loss {:.5f}, reco vs. true: X {:.5f} vs. {:.5f}, PE {:.5f} vs. {:.5f}"""
        print(template.format(track_id, flash_id, correct, loss, true_x, reco_x, true_pe, reco_pe))