import numpy as np
import copy
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlashMatchInput:
    def __init__(self):
        # input array of Flash
        self.flash_v = []
        # input array of QCluster
        self.qcluster_v = []
        # "RAW" QCluster (optional, may not be present, before x-shift)
        self.raw_qcluster_v = []
        # trajectory segment points
        self.track_v = []
        # dx between qcluster and raw_qcluster
        self.x_shift = []
        # True matches, an array of integer-pairs.
        self.true_match = []

class FlashMatch:
    def __init__(self):
        pass

class Flash:
    def __init__(self, *args):
        self.pe_v = []
        self.pe_err_v = []
        self.idx = np.inf    # index from original larlite vector
        self.time = np.inf   # Flash timing, a candidate T0
        self.true_time = np.inf  # MCFlash timing

    def __len__(self):
        return len(self.pe_v)

    def sum(self):
        if len(self.pe_v) == 0:
            return 0
        return torch.sum(self.pe_v)

class QCluster:
    def __init__(self, *args):
        self.qpt_v = []
        self.idx = np.inf # index from original larlite vector
        self.time = np.inf # assumed time w.r.t trigger for reconstruction
        self.true_time = np.inf # time from MCTrack information

    def __len__(self):
        return len(self.qpt_v)

    # shift qcluster_v by given dx
    def shift(self, dx):
        other = copy.deepcopy(self)
        other.qpt_v[:, 0] += dx
        other.xmin += dx
        other.xmax += dx
        return other

    # fill qucluster content from a qcluster_v list
    def fill(self, qpt_v):
        self.qpt_v = torch.tensor(qpt_v, device=device)
        self.xmin = torch.min(self.qpt_v[:, 0]).item()
        self.xmax = torch.max(self.qpt_v[:, 0]).item()

    # drop points outside specified recording range
    def drop(self, x_min, x_max):
        mask = (self.qpt_v[:, 0] >= x_min) & (self.qpt_v[:, 0] <= x_max)
        self.qpt_v = self.qpt_v[mask]