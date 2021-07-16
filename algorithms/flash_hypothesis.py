import numpy as np
import yaml
from utils import Flash

class FlashHypothesis():
    def __init__(self, photon_library, detector_specs, cfg_file=None):
        self.NUM_PROCESS = 4
        self.detector = detector_specs
        self.plib = photon_library
        self.global_qe = 0.0093
        self.qe_v = []  # CCVCorrection factor array
        self.extend_tracks = 0
        self.threshold_extend_tracks = 5.0
        self.segment_size = 0.5
        if cfg_file:
            self.configure(cfg_file)

    def configure(self, cfg_file):
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)["PhotonLibHypothesis"]
        self.global_qe = config["GlobalQE"]
        self.qe_v = config["CCVCorrection"]
        self.extend_tracks = config["ExtendTracks"]
        self.threshold_extend_tracks = config["ThresholdExtendTrack"]
        self.segment_size = config["SegmentSize"]

    def fill_estimate(self, old_track):
        """
        fill flash hypothsis based on given qcluster track
        ---------
        Arguments
          old_track: qcluster track of 3D position + charge
        -------
        Returns
        """
        flash = Flash([])
        track = old_track.copy()
        # extend track if necessary
        if self.extend_tracks:
            min_idx = np.argmin(np.array(track)[:, 0])
            max_idx = np.argmax(np.array(track)[:, 0])

            if ((track[min_idx][0] - self.detector['ActiveVolumeMin'][0] < self.threshold_extend_tracks) or
                (track[min_idx][1] - self.detector['ActiveVolumeMin'][1] < self.threshold_extend_tracks) or
                (track[min_idx][2] - self.detector['ActiveVolumeMin'][2] < self.threshold_extend_tracks) or 
                (self.detector['ActiveVolumeMax'][0] - track[max_idx][0] < self.threshold_extend_tracks) or
                (self.detector['ActiveVolumeMax'][1] - track[max_idx][1] < self.threshold_extend_tracks) or
                (self.detector['ActiveVolumeMax'][2] - track[max_idx][2] < self.threshold_extend_tracks)):

                A = np.array([track[max_idx][0], track[max_idx][1], track[max_idx][2]])
                B = np.array([track[min_idx][0], track[min_idx][1], track[min_idx][2]])
                AB = B - A
                x_C = self.detector['PhotonLibraryVolumeMin'][0]
                lengthAB = np.linalg.norm(AB)
                lengthBC = (x_C - B[0]) / (B[0] - A[0]) * lengthAB
                C = B + AB / lengthAB * lengthBC
                unit = (C - B) / np.linalg.norm(C - B)

                # add to track the part between boundary and C
                num_pts = int(lengthBC / self.segment_size)
                current = np.copy(B)
                for i in range(num_pts+1):
                    current_segment_size = self.segment_size if i < num_pts else (lengthBC - self.segment_size*num_pts)
                    current = current + unit * current_segment_size / 2.0
                    q = current_segment_size * self.detector['LightYield'] * self.detector['MIPdEdx']
                    if (track[0][0] < track[-1][0]):
                        track.insert(0, [current[0], current[1], current[2], q])
                    else:
                        track.append([current[0], current[1], current[2], q])
                    current = current + unit * current_segment_size / 2.0

        # fill estimate
        n_pmt = self.detector['NOpDets']
        local_pe_v = np.zeros(n_pmt)
        for qpt in track:
            local_pe_v += self.plib.VisibilityFromXYZ(qpt[:3])*qpt[3]
        # pool = Pool(self.NUM_PROCESS)
        # loca_pe_v = np.sum(pool.map(self.compute_visibility, track), axis=0)

        for ipmt in range(n_pmt):
            correction = 1. if not self.qe_v else self.qe_v[ipmt]
            flash.append(local_pe_v[ipmt] * self.global_qe / correction)
        
        return flash


    def compute_visibility(self, qpt):
        """
        helper function that computes visibility from a given qcluster point
        ---------
        Arguments
          qpt: qcluster point with 3D positon + charge
        -------
        Returns
          visibility for all PMTs
        """
        return self.plib.VisibilityFromXYZ(qpt[:3])*qpt[3]








