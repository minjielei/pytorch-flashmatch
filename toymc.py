from algorithms.lightpath import LightPath
from algorithms.flashalgo import FlashAlgo
from flashmatch_types import FlashMatchInput, Flash
import numpy as np
import yaml

class ToyMC():
    def __init__(self, photon_library, detector_file='data/detector_specs.yml', cfg_file=None):
        self.detector = yaml.load(open(detector_file), Loader=yaml.Loader)['DetectorSpecs']
        self.plib = photon_library
        self.qcluster_algo = LightPath(self.detector, cfg_file)
        self.flash_algo = FlashAlgo(photon_library, cfg_file)
        self.time_algo = 'random'
        self.track_algo = 'random'
        self.periodTPC = [-1000, 1000]
        self.periodPMT = [-1000, 1000]
        self.ly_variation = 0.0
        self.pe_variation = 0.0
        self.truncate_tpc = 0
        if cfg_file:
            self.configure(cfg_file)

    def configure(self, cfg_file):
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)["ToyMC"]
        self.time_algo = config["TimeAlgo"]
        self.track_algo = config["TrackAlgo"]
        self.periodTPC = config["PeriodTPC"]
        self.periodPMT = config["PeriodPMT"]
        self.ly_variation = config["LightYieldVariation"]
        self.pe_variation = config["PEVariation"]
        self.posx_variation = config['PosXVariation']
        self.truncate_tpc = config["TruncateTPC"]
        self.num_tracks = config["NumTracks"]
        if 'NumpySeed' in config:
            np.random.seed(config['NumpySeed'])

    # create pairs of TPC and PMT matches
    def make_flashmatch_input(self, num_match=None):
        """
        Make N input pairs for flash matching
        --------
        Arguments
        --------
        Returns
         Generated trajectory, tpc, pmt, and raw tpc arrays
        """
        if num_match is None:
            num_match = self.num_tracks

        result = FlashMatchInput()
        
        # generate 3D trajectories inside the detector
        track_v = self.gen_trajectories(num_match)
        result.track_v = track_v
        # generate flash time and x shift (for reco x position assuming trigger time)
        xt_v = self.gen_xt_shift(len(track_v))
        # Defined allowed x recording regions
        min_tpcx, max_tpcx = [t * self.detector['DriftVelocity'] for t in self.periodTPC]
        # generate flash and qclusters
        for idx, track in enumerate(track_v):
            # create raw TPC position and light info
            raw_qcluster = self.make_qcluster(track)
            raw_qcluster.idx = idx
            # Create PMT PE spectrum from raw qcluster
            flash = self.make_flash(raw_qcluster.qpt_v)
            flash.idx = idx
            # Apply x shift and set flash time
            ftime, dx = xt_v[idx]
            flash.time = ftime
            flash.true_time = ftime
            qcluster = raw_qcluster.shift(dx)
            qcluster.idx = idx
            qcluster.true_time = ftime
            raw_qcluster.true_time = ftime
            # Drop qcluster points that are outside the recording range
            if self.truncate_tpc:
                qcluster.drop(min_tpcx, max_tpcx)
            # check for orphan
            valid_match = len(qcluster) > 0 and flash.sum() > 0
            if len(qcluster) > 0:
                result.qcluster_v.append(qcluster)
                result.raw_qcluster_v.append(raw_qcluster)
            if flash.sum() > 0:
                result.flash_v.append(flash)
            if valid_match:
                result.true_match.append((idx,idx))

        return result

    def gen_trajectories(self, num_tracks):
        """
        Generate N random trajectories.
        ---------
        Arguments
          num_tracks: int, number of tpc trajectories to be generated
        -------
        Returns
          a list of trajectories, each is a pair of 3D start and end points
        """
        res = []

        # load detector dimension
        xmin, ymin, zmin = self.detector['ActiveVolumeMin']
        xmax, ymax, zmax = self.detector['ActiveVolumeMax']

        for i in range(num_tracks):
            if self.track_algo=="random":
                start_pt = [np.random.random() * (xmax - xmin) + xmin,
                            np.random.random() * (ymax - ymin) + ymin,
                            np.random.random() * (zmax - zmin) + zmin]
                end_pt = [np.random.random() * (xmax - xmin) + xmin,
                          np.random.random() * (ymax - ymin) + ymin,
                          np.random.random() * (zmax - zmin) + zmin]
            elif self._track_algo=="top-bottom":
                start_pt = [np.random.random() * (xmax - xmin) + xmin,
                            ymax,
                            np.random.random() * (zmax - zmin) + zmin]
                end_pt = [np.random.random() * (xmax - xmin) + xmin,
                          ymin,
                          np.random.random() * (zmax - zmin) + zmin]
            else:
                raise Exception("Track algo not recognized, must be one of ['random', 'top-bottom']")
            res.append([start_pt, end_pt])

        return res

    def gen_xt_shift(self, n):
        """
        Generate flash timing and corresponding X shift
        ---------
        Arguments
          n: int, number of track/flash (number of flash time to be generated)
        -------
        Returns
          a list of pairs, (flash time, dx to be applied on TPC points)
        """
        time_dx_v = []
        duration = self.periodPMT[1] - self.periodPMT[0]
        for idx in range(n):
            t,x=0.,0.
            if self.time_algo == 'random':
                t = np.random.random() * duration + self.periodPMT[0]
            elif self.time_algo == 'periodic':
                t = (idx + 0.5) * duration / n + self.periodPMT[0]
            elif self.time_algo == 'same':
                t = 0.
            else:
                raise Exception("Time algo not recognized, must be one of ['random', 'periodic']")
            x = t * self.detector['DriftVelocity']
            time_dx_v.append((t,x))
        return time_dx_v

    def make_qcluster(self, track):
        """
        Create a qcluster instance from a trajectory
        ---------
        Arguments
          track: trajectory defined by 3D points
        -------
        Returns
          a qcluster instance 
        """
        qcluster = self.qcluster_algo.make_qcluster_from_track(track)
        # apply variation if needed
        if self.ly_variation > 0:
            var = abs(np.random.normal(1.0, self.ly_variation, len(qcluster)))
            for idx in range(len(qcluster)): qcluster.qpt_v[idx][-1] *= var[idx]
        if self.posx_variation > 0:
            var = abs(np.random.normal(1.0, self.posx_variation/qcluster.xsum(), len(qcluster)))
            for idx in range(len(qcluster)): qcluster.qpt_v[idx][0] *= var[idx]

        return qcluster

    def make_flash(self, qcluster):
        """
        Create a flash instance from a qcluster
        ---------
        Arguments
          qcluster: array of 3D position + charge
        -------
        Returns
          a flash instance 
        """
        flash = Flash()
        flash.pe_v = self.flash_algo.fill_estimate(qcluster)
        # apply variation if needed
        var = np.ones(shape=(len(flash)),dtype=np.float32)
        if self.pe_variation>0.:
            var = abs(np.random.normal(1.0,self.pe_variation,len(flash)))
        for idx in range(len(flash)):
            estimate = float(int(np.random.poisson(flash.pe_v[idx].item() * var[idx])))
            flash.pe_v[idx] = estimate
            flash.pe_err_v.append(np.sqrt(estimate))

        return flash


