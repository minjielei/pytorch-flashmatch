import numpy as np
from utils import QCluster

class LightPath():
    def __init__(self, detector_specs):
        self.gap = 0.5
        self.dEdxMIP = detector_specs['MIPdEdx']
        self.light_yield = detector_specs['LightYield']
    
    def fill_qcluster(self, pt1, pt2, qcluster):
        """
        Fill a qcluster instance based given trajectory poins and detector specs
        ---------
        Arguments
          pt1, pt2: 3D trajectory points
          qcluster: qcluster to be filled
          dedx: stopping power
          light yield: light yield
        -------
        Returns
        """
        dist = np.linalg.norm(pt1 - pt2)
        q_pt = []
        # segment less than gap threshold
        if dist < self.gap:
            mid_pt = (pt1 + pt2) / 2 
            q = self.light_yield * self.dEdxMIP * dist
            qcluster.append([mid_pt[0], mid_pt[1], mid_pt[2], q])
            return
        # segment larger than gap threshold
        num_div = int(dist / self.gap)
        direct = (pt1 - pt2) / dist

        for div_idx in range(num_div+1):
            if div_idx < num_div:
                mid_pt = pt2 + direct * (self.gap * div_idx + self.gap / 2.)
                q = self.light_yield * self.dEdxMIP * self.gap
                qcluster.append([mid_pt[0], mid_pt[1], mid_pt[2], q])
            else:
                weight = (dist - int(dist / self.gap) * self.gap)
                mid_pt = pt2 + direct * (self.gap * div_idx + weight / 2.)
                q = self.light_yield * self.dEdxMIP * weight
                qcluster.append([mid_pt[0], mid_pt[1], mid_pt[2], q])


    def make_qcluster_from_track(self, track):
        """
        Create a qcluster instance from a trajectory
        ---------
        Arguments
          track: array of trajectory points
        -------
        Returns
          a qcluster instance
        """
        res = QCluster([])

        # add first point of trajectory
        res.append([track[0][0], track[0][1], track[0][2], 0.])

        for i in range(len(track)-1):
            self.fill_qcluster(np.array(track[i]), np.array(track[i+1]), res)

        # add last point of trajectory
        res.append([track[-1][0], track[-1][1], track[-1][2], 0.])

        return res
