import numpy as np
import copy
import torch
from scipy.optimize import linear_sum_assignment
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FlashMatchInput:
    def __init__(self):
        # input array of Flash
        self.flash_v = []
        # input array of QCluster
        self.qcluster_v = []
        # "RAW" QCluster (optional, may not be present, before x-shift)
        self.raw_qcluster_v = []
        # "RAW" flashmatch::QCluster_t (optional, may not be present, before active BB cut)
        self.all_pts_v = []
        # trajectory segment points
        self.track_v = []
        # True matches, an array of integer-pairs.
        self.true_match = []

class FlashMatch:
    def __init__(self, num_qclusters, num_flashes):
        self.loss_matrix = np.zeros((num_qclusters, num_flashes))
        self.reco_x_matrix = np.zeros((num_qclusters, num_flashes))
        self.reco_pe_matrix = np.zeros((num_qclusters, num_flashes))
    
    def bipartite_match(self):
        row_idx, col_idx = linear_sum_assignment(self.loss_matrix)
        self.tpc_ids = row_idx
        self.flash_ids = col_idx
        self.loss_v = self.loss_matrix[row_idx, col_idx]
        self.reco_x_v = self.reco_x_matrix[row_idx, col_idx]
        self.reco_pe_v = self.reco_pe_matrix[row_idx, col_idx]

class Flash:
    def __init__(self, *args):
        self.pe_v = []
        self.pe_true_v = []
        self.pe_err_v = []
        self.idx = np.inf    # index from original larlite vector
        self.time = np.inf   # Flash timing, a candidate T0
        self.time_true = np.inf  # MCFlash timing
        self.dt_next = np.inf   # dt to next flash
        self.dt_prev = np.inf   # dt to previous flash

    def __len__(self):
        return len(self.pe_v)

    def to_torch(self):
        self.pe_v = torch.tensor(self.pe_v, device=device)
        self.pe_true_v = torch.tensor(self.pe_true_v, device=device)

    def sum(self):
        if len(self.pe_v) == 0:
            return 0
        return torch.sum(self.pe_v).item()

class QCluster:
    def __init__(self, *args):
        self.qpt_v = []
        self.idx = np.inf # index from original larlite vector
        self.time = np.inf # assumed time w.r.t trigger for reconstruction
        self.time_true = np.inf # time from MCTrack information
        self.xshift = 0

    def __len__(self):
        return len(self.qpt_v)

    def __iadd__(self, other):
        if len(self.qpt_v) == 0:
            return other
        else:
            self.qpt_v = torch.cat((self.qpt_v, other.qpt_v), 0)
        return self

    def copy(self):
        return copy.deepcopy(self)

    # total length of the track
    def length(self):
        res = 0
        for i in range(1, len(self.qpt_v)):
            res += torch.linalg.norm(self.qpt_v[i, :3] - self.qpt_v[i-1, :3]).item()
        return res

    # sum over charge 
    def sum(self):
        if len(self.qpt_v) == 0:
            return 0
        return torch.sum(self.qpt_v[:, -1]).item()

    # sum over x coordinates of the track
    def xsum(self):
        if len(self.qpt_v) == 0:
            return 0
        return torch.sum(self.qpt_v[:, 0]).item()

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
    def drop(self, x_min, x_max, y_min = -np.inf, y_max = np.inf, z_min = -np.inf, z_max = np.inf):
        mask = (self.qpt_v[:, 0] >= x_min) & (self.qpt_v[:, 0] <= x_max) & \
            (self.qpt_v[:, 1] >= y_min) & (self.qpt_v[:, 1] <= y_max) & \
            (self.qpt_v[:, 2] >= z_min) & (self.qpt_v[:, 2] <= z_max)
        self.qpt_v = self.qpt_v[mask]