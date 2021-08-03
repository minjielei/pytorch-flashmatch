import numpy as np
import torch
import copy
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

    def make_torch_input(self):
        target = torch.tensor(self.flash_v, device=device)
        input = [torch.tensor(qcluster, device=device) for qcluster in self.qcluster_v]
        return input, target

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

    # return the total trajectory length
    def length(self):
        res = 0
        for i in range(1, len(self)):
            res += np.linalg.norm(self[i] - self[i-1]) 
        return res

    # drop points outside specified range
    def drop(self, x_min, x_max, y_min = -np.inf, y_max = np.inf, z_min = -np.inf, z_max = np.inf):
        old_qcluster = self.copy()
        self.clear()
        for pt in old_qcluster:
            if pt[0] < x_min: continue
            if pt[0] > x_max: continue
            if pt[1] < y_min: continue
            if pt[1] > y_max: continue
            if pt[2] < z_min: continue
            if pt[2] > z_max: continue
            self.append(pt)

def extend_track(detector, threshold, segment_size, track):
    """
    extend a qcluster track if it is within threshold to detector boundary
    """
    min_idx = np.argmin(np.array(track)[:, 0])
    max_idx = np.argmax(np.array(track)[:, 0])

    if ((track[min_idx][0] - detector['ActiveVolumeMin'][0] < threshold) or
        (track[min_idx][1] - detector['ActiveVolumeMin'][1] < threshold) or
        (track[min_idx][2] - detector['ActiveVolumeMin'][2] < threshold) or 
        (detector['ActiveVolumeMax'][0] - track[max_idx][0] < threshold) or
        (detector['ActiveVolumeMax'][1] - track[max_idx][1] < threshold) or
        (detector['ActiveVolumeMax'][2] - track[max_idx][2] < threshold)):

        A = np.array([track[max_idx][0], track[max_idx][1], track[max_idx][2]])
        B = np.array([track[min_idx][0], track[min_idx][1], track[min_idx][2]])
        AB = B - A
        x_C = detector['PhotonLibraryVolumeMin'][0]
        lengthAB = np.linalg.norm(AB)
        lengthBC = (x_C - B[0]) / (B[0] - A[0]) * lengthAB
        C = B + AB / lengthAB * lengthBC
        unit = (C - B) / np.linalg.norm(C - B)

        # add to track the part between boundary and C
        num_pts = int(lengthBC / segment_size)
        current = np.copy(B)
        for i in range(num_pts+1):
            current_segment_size = segment_size if i < num_pts else (lengthBC - segment_size*num_pts)
            current = current + unit * current_segment_size / 2.0
            q = current_segment_size * detector['LightYield'] * detector['MIPdEdx']
            if (track[0][0] < track[-1][0]):
                track.insert(0, [current[0], current[1], current[2], q])
            else:
                track.append([current[0], current[1], current[2], q])
            current = current + unit * current_segment_size / 2.0

def get_x_constraints(input, detector_specs):
    """
    compute the x shift bounds of given qcluster input for it to be within active volume
    """
    vol_x_min = detector_specs['ActiveVolumeMin'][0]
    vol_x_max = detector_specs['ActiveVolumeMax'][0]
    track_x_min, track_x_max = torch.min(input[:, 0]).item(), torch.max(input[:, 0]).item()
    x_min = min(vol_x_min - track_x_min, vol_x_max - track_x_max, 0)
    x_max = max(vol_x_min - track_x_min, vol_x_max - track_x_max)
    return x_min, x_max
    