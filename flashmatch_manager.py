import numpy as np
from toymc import ToyMC

class FlashMatchManager():
    """
    Top level FlashMatchManager program that runs the algorithm
    """
    def __init__(self, detector_cfg, photon_library, cfg):
        self.configure(detector_cfg, photon_library, cfg)

    def configure(self, detector_cfg, photon_library, cfg):
        self.reader = ToyMC(detector_cfg, photon_library, cfg)
        self.hypothesis_algo = ToyMC.flash_algo
        self.match_algo = None

    def make_flashmatch_input(self, num_tracks):
        pass

    def flash_hypothesis(self,qcluster):
        pass

    def train(self):
        pass

    def inference(self):
        pass

    def run_flash_match(self,match_input):
        pass