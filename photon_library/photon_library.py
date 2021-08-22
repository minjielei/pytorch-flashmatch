import h5py  as h5
import numpy as np
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PhotonLibrary(object):
    def __init__(self, fname='photon_library/plib_20201209.h5', pmt_loc='photon_library/pmt_loc.csv'):
        if not os.path.isfile(fname):
            print('Downloading photon library file... (>300MByte, may take minutes')
            os.system('curl -O https://www.nevis.columbia.edu/~kazuhiro/plib.h5 ./')
        if not os.path.isfile(fname):
            print('Error: failed to download the photon library file...')
            raise Exception

        with h5.File(fname,'r') as f:
            self._vis  = torch.from_numpy(np.array(f['vis'])).to(device)
            self._min  = torch.tensor(f['min']).to(device)
            self._max  = torch.tensor(f['max']).to(device)
            self.shape = torch.tensor(f['numvox']).to(device)

        self.gap = (self._max[0] - self._min[0]) / self.shape[0] # x distance between adjacent voxels

    def VisibilityFromXYZ(self, pos, ch=None):
        if not torch.is_tensor(pos):
          pos = torch.tensor(pos, device=device)
        return self.Visibility(self.Position2VoxID(pos), ch)

    def Visibility(self, vids, ch=None):
        '''
        Returns a probability for a detector to observe a photon.
        If ch (=detector ID) is unspecified, returns an array of probability for all detectors
        INPUT
          vids - Tensor of integer voxel IDs
          ch  - Integer (valid range 0 to N-1) to specify an optical detector (optional)
        RETURN
          Probability(ies) in FP32 to observe a photon at a specified location for each vid
        '''
        if ch is None:
            return self._vis[vids]
        return self._vis[vids][ch]

    def Position2VoxID(self, pos):
        '''
        Takes a tensor of xyz position (x,y,z) and converts to a tensor of voxel IDs
        INPUT
          pos - Tensor of length 3 floating point array noting the position along xyz axis
        RETURN
          Tensor of sigle integer voxel IDs       
        '''
        axis_ids = ((pos - self._min) / (self._max - self._min) * self.shape).int()

        return (axis_ids[:, 0] + axis_ids[:, 1] * self.shape[0] +  axis_ids[:, 2]*(self.shape[0] * self.shape[1])).long()
