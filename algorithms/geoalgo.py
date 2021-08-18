import numpy as np

class GeoAlgo:
    def __init__(self, detector_specs):
        self.vol_min = detector_specs["ActiveVolumeMin"]
        self.vol_max = detector_specs["ActiveVolumeMax"]
        self.unit_factor = 0.0001

    def box_overlap(self, traj):
        """
        Return the overlapping segments between given trajectory and the detector
        ---------
        Arguments
          traj: array of trajectory points
        -------
        Returns 
          new trajectory segments that are inside the detector
        """
        if len(traj) == 0:
            return traj
        # if first & last points inside, then return full trajectory
        if self.contain(traj[0]) and self.contain(traj[-1]):
            return traj

        res = []
        for i in range(len(traj)-1):
            pt0, pt1 = traj[i], traj[i+1]
            if self.contain(pt0): res.append(pt0)
            # Are we stepping through the boundary, need to account for both directions of trajectory
            if (self.contain(pt0) and not self.contain(pt1)) or (not self.contain(pt1) and self.contain(pt1)):
                unit = (pt1 - pt0) / np.linalg.norm(pt1 - pt0)
                pt0, pt1 = pt0 - self.unit_factor * unit, pt1 + self.unit_factor * unit
                crossing_pts = self.intersection(pt0, pt1)

                for pt in crossing_pts:
                    if self.contain(pt): res.append(pt)
        if self.contain(traj[-1]): res.append(traj[-1])

        return np.array(res)

    def intersection(self, start_pt, end_pt):
        """
        Get the intersection points between line and detector
        ---------
        Arguments
          start_pt: start point of line
          end_ptL end point of line 
        -------
        Returns 
          list of intersection points
        """
        res = []
        xs1 = np.array([np.inf, np.inf, np.inf])
        xs2 = np.array([np.inf, np.inf, np.inf])

        min_plane_n = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        max_plane_n = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        line_dir = end_pt - start_pt

        # check the case of parallel line
        for i in range(len(self.vol_min)):
            if (line_dir[i] == 0 and start_pt[i] < self.vol_min[i] or self.vol_max[i] <= start_pt[i]):
                return res
        
        # look for xs with 3 min planes
        for i in range(3):
            s = -1 * np.dot(min_plane_n[i], (start_pt - self.vol_min)) / np.dot(min_plane_n[i], line_dir)
            if s < 0: continue
            xs = start_pt + line_dir * s
            # check if the found pt is within the surface area of other 2 axes
            on_surface = True
            for sur_axis in range(3):
                if sur_axis == i: continue
                if (xs[sur_axis] < self.vol_min[sur_axis] or self.vol_max[sur_axis] < xs[sur_axis]):
                    on_surface = False
                    break
            if (on_surface and (xs != xs1).any()):
                if (xs1 == np.inf).any(): 
                    np.copyto(xs1, xs)
                else:
                    np.copyto(xs2, xs)
                    break
        # if xs2 is filled, simply return the result. Order the output via distance
        if (xs2 != np.inf).all():
            if (np.linalg.norm(xs1 - start_pt) <= np.linalg.norm(xs2 - start_pt)):
                res.append(xs2)
                res.append(xs1)
            else:
                res.append(xs1)
                res.append(xs2)
            return res
        
        # look for xs with 3 max planes
        for i in range(3):
            s = -1 * np.dot(max_plane_n[i], (start_pt - self.vol_max)) / np.dot(max_plane_n[i], line_dir)
            if s < 0: continue
            xs = start_pt + line_dir * s
            # check if the found pt is within the surface area of other 2 axes
            on_surface = True
            for sur_axis in range(3):
                if sur_axis == i: continue
                if (xs[sur_axis] < self.vol_min[sur_axis] or self.vol_max[sur_axis] < xs[sur_axis]):
                    on_surface = False
                    break
            if (on_surface and (xs != xs1).any()):
                if (xs1 == np.inf).any(): 
                    np.copyto(xs1, xs)
                else:
                    np.copyto(xs2, xs)
                    break
        if (xs1 == np.inf).any(): return res
        if (xs2 != np.inf).all():
            if (np.linalg.norm(xs1 - start_pt) <= np.linalg.norm(xs2 - start_pt)):
                res.append(xs2)
                res.append(xs1)
            else:
                res.append(xs1)
                res.append(xs2)
            return res
        res.append(xs1)
        return res

    def contain(self, pt):
        """
        check if a pt is inside detector active volume
        ---------
        Arguments
          pt: 3d pt as numpy array 
        -------
        Returns 
          True if pt is inside detector, false otherwise
        """
        return not((pt[0] < self.vol_min[0] or self.vol_max[0] < pt[0]) or \
            (pt[1] < self.vol_min[1] or self.vol_max[1] < pt[1]) or \
            (pt[2] < self.vol_min[2] or self.vol_max[2] < pt[2]))
