import h5py  as h5
import numpy as np
import os
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PhotonLibrary(object):
    def __init__(self, fname='photon_library/plib_20180801.h5', pmt_loc='photon_library/pmt_loc.csv'):
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

        self.gap = 5.0 # distance between adjacent voxels
        pmt_data = np.loadtxt(pmt_loc,skiprows=1,delimiter=',')
        if not (pmt_data[:,0].astype(np.int32) == np.arange(pmt_data.shape[0])).all():
            raise Exception('pmt_loc.csv contains optical channel data not in order of channel numbers')
        self._pmt_pos = torch.from_numpy(pmt_data[:,1:4]).to(device)
        self._pmt_dir = torch.from_numpy(pmt_data[:,4:7]).to(device)
        if not self._pmt_pos.shape[0] == self._vis.shape[1]:
            raise Exception('Optical detector count mismatch: photon library %d v.s. pmt_loc.csv %d' % (self._vis.shape[1],
                                                                                                        self._pmt_pos.shape[0]))
        # Convert the PMT positions in a normalized coordinate (fractional position within the voxelized volume)
        self._pmt_pos = (self._pmt_pos - self._min) / (self._max - self._min)

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
