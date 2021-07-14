from algorithms.lightpath import LightPath
from algorithms.flash_hypothesis import FlashHypothesis
from utils import x_shift, truncate_qcluster
import numpy as np
import yaml
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ToyMC():
    def __init__(self, detector_cfg, photon_library):
        self.detector = yaml.safe_load(detector_cfg)['DetectorSpecs']
        self.plib = photon_library
        self.qcluster_algo = LightPath(self.detector)
        self.flash_algo = FlashHypothesis(self.detector, photon_library)
        self.time_algo = 'random'
        self.track_algo = 'random'
        self.periodTPC = [-1000, 1000]
        self.periodPMT = [-1000, 1000]
        self.ly_variation = 0.0
        self.pe_variation = 0.0
        self.truncate_tpc = 0

    # create pairs of TPC and PMT matches
    def make_flashmatch_input(self, num_match=None, use_numpy=True):
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

        array_ctor = np.array if use_numpy else torch.Tensor

        pmt_v = []
        tpc_v = []
        raw_tpc_v = []
        true_match = []
        
        # generate 3D trajectories inside the detector
        track_v = self.gen_trajectories(num_match)
        # generate flash time and x shift (for reco x position assuming trigger time)
        xt_v = self.gen_xt_shift(len(track_v))
        # Defined allowed x recording regions
        mintpcx, max_tpcx = [t * self.detector['DriftVelocity'] for t in self.periodTPC]
        # generate flash and qclusters
        for idx, track in enumerate(track_v):
            # create raw TPC position and light info
            raw_qcluster = self.make_qcluster(track)
            # Create PMT PE spectrum from raw qcluster
            flash = self.make_flash(raw_qcluster)
            # Apply x shift and set flash time
            ftime, dx = xt_v[idx]
            qcluster = x_shift(raw_qcluster, dx)
            # Drop qcluster points that are outside the recording range
            if self.truncate_tpc:
                qcluster = truncate_qcluster(qcluster)
            # check for orphan
            valid_match = len(qcluster) > 0 and np.sum(flash) > 0
            if len(qcluster) > 0:
                tpc_v.append(array_ctor(qcluster))
                raw_tpc_v.append(array_ctor(raw_qcluster))
            if np.sum(flash) > 0:
                pmt_v.append(array_ctor(flash))
            if valid_match:
                true_match.append((idx,idx))

        return track_v, tpc_v, pmt_v, raw_tpc_v

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
            var = abs(np.random.normal(1.0, self.ly_variation, qcluster.size()))
            for idx in range(qcluster.size()): qcluster[idx][-1] *= var[idx]

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
        flash = self.flash_algo.fill_estimate(qcluster)
        # apply variation if needed
        var = np.ones(shape=(len(flash)),dtype=np.float32)
        if self.pe_variation>0.:
            var = abs(np.random.normal(1.0,self.pe_variation,len(flash)))
        for idx in range(len(flash)):
            estimate = float(int(np.random.poisson(flash[idx] * var[idx])))
            flash[idx] = estimate

        return flash


