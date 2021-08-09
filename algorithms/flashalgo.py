import numpy as np
import torch
import yaml
from flashmatch_types import Flash
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.qe_v = torch.tensor(config["CCVCorrection"], device=device)

    def fill_estimate(self, track, use_tensor=False):
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
        if not torch.is_tensor(track):
          track = torch.tensor(track, device=device)
        local_pe_v = torch.sum(self.plib.VisibilityFromXYZ(track[:, :3])*(track[:, 3].unsqueeze(-1)), axis = 0)

        if len(self.qe_v) == 0:
          self.qe_v = torch.ones(local_pe_v.shape, device=device)
        res = local_pe_v * self.global_qe * self.qe_v
        if use_tensor:
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
        vids = self.plib.Position2VoxID(track[:, :3])
        grad = (self.plib.Visibility(vids+1) - self.plib.Visibility(vids)) / self.plib.gap
        return grad * (track[:, 3].unsqueeze(-1)) * self.global_qe * self.qe_v