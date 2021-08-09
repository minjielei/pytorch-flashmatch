import numpy as np
import copy

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

class Flash(list):
    def __init__(self, *args):
        super(Flash, self).__init__(args[0])
        self.pe_err_v = []
        self.idx = np.inf    # index from original larlite vector
        self.time = np.inf   # Flash timing, a candidate T0
        self.true_time = np.inf  # MCFlash timing
        

class QCluster(list):
    def __init__(self, *args):
        super(QCluster, self).__init__(args[0]) # collection of charge deposition 3D points
        self.idx = np.inf # index from original larlite vector
        self.time = 0 # assumed time w.r.t trigger for reconstruction
        self.true_time = np.inf # time from MCTrack information

    def __add__(self, shift):
        rhs = copy.deepcopy(self)
        for i in range(len(self)):
            rhs[i][0] += shift
        return rhs

    def __iadd__(self, shift):
        for i in range(len(self)):
            self[i][0] += shift
        return self
    
    def __sub__(self, shift):
        rhs = copy.copy(self)
        for i in range(len(self)):
            rhs[i][0] -= shift
        return rhs

    def __isub__(self, shift):
        for i in range(len(self)):
            self[i][0] -= shift
        return self

    # return min and maximum x points of the qcluster
    def x_min_max(self):
        track = np.array(self)
        return np.min(track[:, 0]), np.max(track[:, 0])

    # return the total trajectory length
    def length(self):
        res = 0
        track = np.array(self)
        for i in range(1, len(track)):
            res += np.linalg.norm(track[i] - track[i-1]) 
        return res

    # drop points outside specified recording range
    def drop(self, x_min, x_max):
        old_qcluster = self.copy()
        self.clear()
        for pt in old_qcluster:
            if pt[0] < x_min: continue
            if pt[0] > x_max: continue
            self.append(pt)