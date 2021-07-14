import numpy as np

class Flash(object):
    def __int__(self):
        self.pe_v = [] # PE distribution over photo-detectors
        self.pe_true_v = [] # PE distribution over photo-detectors of MCFlash
        self.pe_err_v = [] # PE value error
        self.pos = [] # Flash position
        self.pos_err = [] # Flash position error
        self.time = np.inf # Flash timing, a candidate T0
        self.time_true = np.inf # MCFlash timing, (if it was matched to a MCFlash)
        self.dt_next = np.inf
        self.dt_prev = np.inf
        self.idx = np.inf # index from original larlite vector

class QCluster(object):
    def __init__(self):
        self.qcluster = [] # collection of charge deposition 3D points
        self.idx = np.inf # index from original larlite vector
        self.time = 0 # assumed time w.r.t trigger for reconstruction
        self.time_true = np.inf # time from MCTrack information

    def __add__(self, shift):
        for i in range(len(self.qcluster)):
            self.qcluster[i][0] += shift
        return self
    
    def __sub__(self, shift):
        for i in range(len(self.qcluster)):
            self.qcluster[i][0] -= shift
        return self

    def append(self, other):
        self.qcluster += other.qcluster

    # sum over charge of every point in cluster
    def sum(self):
        if not self.qcluster:
            return 0
        return np.sum(np.array(self.qcluster)[:, -1])

    # get the minimum x from all cluster points
    def min_x(self):
        if not self.qcluster:
            return 0
        return np.min(np.array(self.qcluster)[:, 0])

    # get the maximum x from all cluster points
    def max_x(self):
        return np.max(self.qcluster[:, 0])

    # return the total trajectory length
    def length(self):
        res = 0
        for i in range(1, len(self.qcluster)):
            res += np.linalg.norm(self.qcluster[i] - self.qcluster[i-1]) 
        return res

    # drop points outside specified range
    def drop(self, x_min, x_max, y_min = -np.inf, y_max = np.max, z_min = -np.inf, z_max = np.inf):
        new_cluster = []
        for idx, pt in enumerate(self.qcluster):
            if pt[0] < x_min: continue
            if pt[0] > x_max: continue
            if pt[1] < y_min: continue
            if pt[1] > y_max: continue
            if pt[2] < z_min: continue
            if pt[2] > z_max: continue
            new_cluster.append(pt)
        self.qcluster = new_cluster

def x_shift(qcluster, shift):
    """
    Shift x positon of given qcluster
    ---------
    Arguments
        qcluster: cluster of 3D points + charge
        shift: amount to shift in x direction
    -------
    Returns
        shifter qcluster
    """
    new_cluster = qcluster.copy()
    for i in range(len(new_cluster)):
        new_cluster[i][0] += shift
    return new_cluster

def truncate_qcluster(qcluster, x_min, x_max):
    """
    Drop points in qcluster that are outside the tpc boundary
    ---------
    Arguments
        qcluster: qcluster to be truncated
        x_min, x_max: boundary of tpc
    -------
    Returns
        truncated qcluster
    """
    new_cluster = []
    for idx, pt in enumerate(qcluster):
        if pt[0] < x_min: continue
        if pt[0] > x_max: continue
        new_cluster.append(pt)
    return new_cluster
    