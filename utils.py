import numpy as np
import copy

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
        rhs = copy.copy(self)
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

    # return the total trajectory length
    def length(self):
        res = 0
        for i in range(1, len(self)):
            res += np.linalg.norm(self[i] - self[i-1]) 
        return res

    # drop points outside specified range
    def drop(self, x_min, x_max, y_min = -np.inf, y_max = np.max, z_min = -np.inf, z_max = np.inf):
        old_qcluster = self.copy()
        self.clear()
        for idx, pt in enumerate(old_qcluster):
            if pt[0] < x_min: continue
            if pt[0] > x_max: continue
            if pt[1] < y_min: continue
            if pt[1] > y_max: continue
            if pt[2] < z_min: continue
            if pt[2] > z_max: continue
            self.append(pt)

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
    