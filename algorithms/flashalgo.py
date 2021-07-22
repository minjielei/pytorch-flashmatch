import numpy as np
import yaml
from utils import Flash

class FlashAlgo():
    def __init__(self, photon_library, cfg_file=None):
        self.plib = photon_library
        self.global_qe = 0.0093
        self.qe_v = []  # CCVCorrection factor array
        if cfg_file:
          self.configure(cfg_file)

    def configure(self, cfg_file):
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)["PhotonLibHypothesis"]
        self.global_qe = config["GlobalQE"]
        self.qe_v = np.array(config["CCVCorrection"])

    def fill_estimate(self, track, use_numpy=False):
        """
        fill flash hypothsis based on given qcluster track
        ---------
        Arguments
          track: qcluster track of 3D position + charge
        -------
        Returns
          a hypothesis Flash object
        """
        # fill estimate
        local_pe_v = self.plib.VisibilityFromXYZ(track[0][:3])*track[0][3]
        for i in range(1, len(track)):
          if track[i][3] != 0:
            local_pe_v += self.plib.VisibilityFromXYZ(track[i][:3])*track[i][3]

        if len(self.qe_v) == 0:
          self.qe_v = np.ones(local_pe_v.shape)
        res = local_pe_v * self.global_qe * self.qe_v
        if use_numpy:
          return res
        
        return Flash(res.tolist())

    def backward_gradient(self, track):
        """
        Compue the gradient of the fill_estimate step for given track
        ---------
        Arguments
          track: qcluster track of 3D position + charge
        -------
        Returns
          gradient value of the fill_estimate step for track
        """
        num_voxel_x = self.plib.shape[0]
        res = []
        for qpt in track: 
          x, y, z, q = qpt
          if q == 0:
            res.append(np.zeros(self.plib.num_pmt))
          else:
            vid = self.plib.Position2VoxID([x, y, z])
            grad = 0
            if vid % num_voxel_x == 0:
              gap = self.plib.VoxID2Position(vid+1)[0] - self.plib.VoxID2Position(vid)[0]
              grad = (self.plib.Visibility(vid+1) - self.plib.Visibility(vid)) / gap
            elif vid % num_voxel_x == 1:
              gap = self.plib.VoxID2Position(vid+1)[0] - self.plib.VoxID2Position(vid-1)[0]
              grad = (self.plib.Visibility(vid+1) - self.plib.Visibility(vid-1)) / gap
            else:
              gap = self.plib.VoxID2Position(vid)[0] - self.plib.VoxID2Position(vid-1)[0]
              grad = (self.plib.Visibility(vid) - self.plib.Visibility(vid-1)) / gap
            res.append(grad * q * self.global_qe * self.qe_v)
        return res