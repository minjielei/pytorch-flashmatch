import numpy as np
from root_numpy import root2array

class ROOTInput:
    def __init__(self, particleana, opflashana, detector_specs, cfg_file=None):
        self.detector = detector_specs
        self.periodTPC = [-1000, 1000]