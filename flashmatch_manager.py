import numpy as np
import torch
import yaml
import itertools
from toymc import ToyMC
from flashmatch_types import FlashMatch
from algorithms.match_model import GradientModel, PoissonMatchLoss, EarlyStopping
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
        self.min_lr = config['MinLearningRate']
        self.scheduler_factor = config['SchedulerFactor']
        self.stopping_patience = config['StoppingPatience']
        self.stopping_delta = config['StoppingDelta']
        self.num_processes = config['NumProcesses']
        self.loss_threshold = config['LossThreshold']

        self.reader = ToyMC(photon_library, detector_file, cfg)
        self.flash_algo = self.reader.flash_algo
        self.loss_fn = PoissonMatchLoss()

        self.vol_xmin = self.detector_specs["ActiveVolumeMin"][0]
        self.vol_xmax = self.detector_specs["ActiveVolumeMax"][0]
        self.drift_velocity = self.detector_specs["DriftVelocity"]
        self.time_shift = config['BeamTimeShift']
        self.touching_track_window = config['TouchingTrackWindow']
        self.offset = config['Offset']

    def make_flashmatch_input(self, num_tracks):
        """
        Make flash matching input using configured reader
        --------
        Arguments
          num_tracks: number of tracks to generate
        --------
        Returns
          FlashMatchInput object
        """
        return self.reader.make_flashmatch_input(num_tracks)

    def match(self, flashmatch_input):
        """
        Run flash matching on flashmatch input
        --------
        Arguments
          flashmatch_input: FlashMatchInput object
        --------
        Returns
          FlashMatch object storing the result of the match
        """
        match = FlashMatch(len(flashmatch_input.qcluster_v), len(flashmatch_input.flash_v))
        paramlist = list(itertools.product(flashmatch_input.qcluster_v, flashmatch_input.flash_v))

        import torch.multiprocessing as mp
        from multiprocessing.pool import ThreadPool
        ctx = mp.get_context("spawn")
        with ThreadPool(processes=self.num_processes) as pool:
            for idx, (loss, reco_x, reco_pe) in enumerate(pool.imap(self.one_pmt_match, paramlist)):
                track_id = paramlist[idx][0].idx
                flash_id = paramlist[idx][1].idx
                match.loss_matrix[track_id, flash_id] = loss
                match.reco_x_matrix[track_id, flash_id] = reco_x
                match.reco_pe_matrix[track_id, flash_id] = reco_pe
        match.bipartite_match()
        return match

    def one_pmt_match(self, params):
        """
        Run flash matching on for one pair of qcluster and flash input
        --------
        Arguments
          params: tuple of (qcluster, flash)
        --------
        Returns
          loss, reco_x, reco_pe
        """
        qcluster, flash = params
        track_xmin, track_xmax = qcluster.xmin, qcluster.xmax
        dx_min, dx_max = self.vol_xmin - track_xmin, self.vol_xmax - track_xmax
        dx0 = - (flash.time - self.time_shift) * self.drift_velocity

        tolerence = self.touching_track_window/2. * self.drift_velocity
        contained_tpc0 = (dx0>=dx_min-tolerence) and (dx0<=dx_max+tolerence)
        contained_tpc1 = (-dx0>=dx_min-tolerence) and (-dx0<=dx_max+tolerence)
        # Inspect, in either assumption (original track is in tpc0 or tpc1), the track is contained in the whole active volume or not
        if contained_tpc0:
            dx0 = max(dx0, self.vol_xmin - track_xmin + self.offset)
        elif contained_tpc1:
            dx0 = min(-dx0, self.vol_xmax - track_xmax - self.offset)
        else:
            return np.inf, np.inf, np.inf

        input = qcluster.qpt_v
        target = flash.pe_v
        return self.train(input, target, dx0, dx_min, dx_max)

    def train(self, input, target, dx0, dx_min, dx_max):
        """
        Run gradient descent model on input
        --------
        Arguments
          input: qcluster input as tensor
          target: flash target as tensor
          dx0: initial xshift in cm
          dx_min: miminal allowed value of dx in cm
          dx_max: maximum allowed value of dx in cm
        --------
        Returns
          loss, reco_x, reco_pe
        """
        model = GradientModel(self.flash_algo, dx0, dx_min, dx_max)
        model.to(device)
        
        # from torch.utils.tensorboard import SummaryWriter
        # writer = SummaryWriter()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=self.min_lr, factor=self.scheduler_factor)
        early_stopping = EarlyStopping(self.stopping_patience, self.stopping_delta)

        for i in range(self.max_iteration):
            pred = model(input)
            loss = self.loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            early_stopping(loss)

            # writer.add_scalar("Loss", loss, i)
            # writer.add_scalar("Reco X", model.xshift.dx-true_x, i)
            # writer.add_scalar("Reco PE - True PE", torch.sum(pred) - torch.sum(target[match.item()]), i)
            if loss > self.loss_threshold or early_stopping.early_stop:
                # print("stopped at iteration ", i)
                break

        return loss.item(), model.xshift.dx.item(), torch.sum(pred).item()

    # compute initial loss on flashmatch_input to study loss separation
    def initial_loss(self, flashmatch_input):
        true_loss = []
        paramlist = list(itertools.product(flashmatch_input.qcluster_v, flashmatch_input.flash_v))
        for qcluster, flash in paramlist:
            input = qcluster.qpt_v
            target = flash.pe_v
            track_xmin, track_xmax = qcluster.xmin, qcluster.xmax
            dx_min, dx_max = self.vol_xmin - track_xmin, self.vol_xmax - track_xmax
            dx0 = - (flash.time - self.time_shift) * self.drift_velocity
            if dx0 >= dx_min and dx0 <= dx_max:
                if qcluster.idx == flash.idx:
                    true_loss.append(self.train_one_step(input, target, dx0, dx_min, dx_max))
        return true_loss

    # train model for one step on flashmatch input to get the initial loss
    def train_one_step(self, input, target, dx0, dx_min, dx_max):
        model = GradientModel(self.flash_algo, dx0, dx_min, dx_max)
        model.to(device)

        pred = model(input)
        loss = self.loss_fn(pred, target)
        return loss.item()
