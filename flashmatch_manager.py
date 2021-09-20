import numpy as np
import torch
import yaml
import itertools
from toymc import ToyMC
from rootinput import ROOTInput
from flashmatch_types import FlashMatch
from algorithms.match_model import GradientModel, PoissonMatchLoss, EarlyStopping
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlashMatchManager():
    """
    Top level FlashMatchManager program that runs the io and matching algorithm
    """
    def __init__(self, detector_cfg, cfg, particleana=None, opflashana=None, photon_library=None):
        self.configure(detector_cfg, cfg, particleana, opflashana, photon_library)

    def configure(self, detector_file, cfg, particleana, opflashana, photon_library):
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

        if particleana is None or opflashana is None:
          self.reader = ToyMC(photon_library, detector_file, cfg)
        else:
          self.reader = ROOTInput(particleana, opflashana, photon_library, detector_file, cfg)
        self.flash_algo = self.reader.flash_algo
        self.loss_fn = PoissonMatchLoss()

        self.vol_xmin = self.detector_specs["ActiveVolumeMin"][0]
        self.vol_xmax = self.detector_specs["ActiveVolumeMax"][0]
        self.tpc0_xmin = self.detector_specs["MinPosition_TPC0"][0]
        self.tpc0_xmax = self.detector_specs["MaxPosition_TPC0"][0]
        self.tpc1_xmin = self.detector_specs["MinPosition_TPC1"][0]
        self.tpc1_xmax = self.detector_specs["MaxPosition_TPC1"][0]

        self.drift_velocity = self.detector_specs["DriftVelocity"]
        self.time_shift = config['BeamTimeShift']
        self.touching_track_window = config['TouchingTrackWindow']
        self.offset = config['Offset']

        self.exp_frac_v = config['PhotonDecayFractions']
        self.exp_tau_v = config['PhotonDecayTimes']

    def entries(self):
        from rootinput import ROOTInput
        if not isinstance(self.reader, ROOTInput):
            return []
        return np.arange(len(self.reader))

    def event_id(self, entry_id):
        from rootinput import ROOTInput
        if not isinstance(self.reader, ROOTInput):
            return -1
        return self.reader.event_id(entry_id)

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
        track_id, flash_id = 0, 0
        with ThreadPool(processes=self.num_processes) as pool:
            for loss, reco_x, reco_pe in pool.imap(self.one_pmt_match, paramlist):
                match.loss_matrix[track_id, flash_id] = loss
                match.reco_x_matrix[track_id, flash_id] = reco_x
                match.reco_pe_matrix[track_id, flash_id] = reco_pe
                if flash_id < len(flashmatch_input.flash_v) - 1:
                  flash_id += 1
                else:
                  track_id += 1
                  flash_id = 0
        match.local_match()
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
        res = []
        qcluster, flash = params
        
        dx0_v, dx_min, dx_max = self.calculate_dx0(flash, qcluster)
        if len(dx0_v) == 0:
          return np.inf, np.inf, np.inf

        # calculate the integral factor to reweight flash based on its time width
        integral_factor = 0
        for i in range(len(self.exp_frac_v)):
            integral_factor += self.exp_frac_v[i] * (1 - np.exp(-1 * flash.time_width / self.exp_tau_v[i]))

        input = qcluster.qpt_v
        target = flash.pe_v / integral_factor

        min_loss = np.inf
        for dx_0 in dx0_v:
            loss, reco_x, reco_pe = self.train(input, target, dx_0, dx_min, dx_max)
            if loss < min_loss:
                min_loss = loss
                res = [loss, reco_x, reco_pe]
        return res

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

    # determine initial dx0 to use for model, assuming track is contained in tpc0 or tpc1
    def calculate_dx0(self, flash, qcluster):
        x0_v = []

        track_xmin, track_xmax = qcluster.xmin, qcluster.xmax
        dx_min, dx_max = self.vol_xmin - track_xmin, self.vol_xmax - track_xmax
        dx0 = (flash.time - self.time_shift) * self.drift_velocity

        # determine initial x0
        tolerence = self.touching_track_window/2. * self.drift_velocity
        contained_tpc0 = (-dx0>=dx_min-tolerence) and (-dx0<=dx_max+tolerence)
        contained_tpc1 = (dx0>=dx_min-tolerence) and (dx0<=dx_max+tolerence)
        # Inspect, in either assumption (original track is in tpc0 or tpc1), the track is contained in the whole active volume or not
        if contained_tpc0:
            x0_v.append(max(-dx0, self.vol_xmin - track_xmin + self.offset))
        if contained_tpc1:
            x0_v.append(min(dx0, self.vol_xmax - track_xmax - self.offset))

        return x0_v, dx_min, dx_max

    # compute initial loss on flashmatch_input to study loss separation
    def initial_loss(self, flashmatch_input):
        true_loss = []
        paramlist = list(itertools.product(flashmatch_input.qcluster_v, flashmatch_input.flash_v))
        for qcluster, flash in paramlist:
            input = qcluster.qpt_v
            target = flash.pe_v

            dx0_v, dx_min, dx_max = self.calculate_dx0(flash, qcluster)

            if (flash.idx, qcluster.idx) in flashmatch_input.true_match:
                res = []
                for dx0 in dx0_v:
                    res.append([self.train_one_step(input, target, dx0, dx_min, dx_max)])
                true_loss.append(min(res))
        return true_loss

    # train model for one step on flashmatch input to get the initial loss
    def train_one_step(self, input, target, dx0, dx_min, dx_max):
        model = GradientModel(self.flash_algo, dx0, dx_min, dx_max)
        model.to(device)

        pred = model(input)
        loss = self.loss_fn(pred, target)
        return loss.item()