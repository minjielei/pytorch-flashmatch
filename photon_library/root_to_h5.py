import uproot
import numpy as np
import h5py
import uproot
import yaml

class Converter(object):
    def __init__(self, det_file='../data/detector_specs.yml', root_file = 'PhotonLibrary-20201209.root'):
        self.detector = yaml.load(open(det_file), Loader=yaml.Loader)['DetectorSpecs']
        self.root_file = root_file
        self.nchannels = self.detector['PhotonLibraryNOpDetChannels']
        self.nvoxels = np.array(self.detector['PhotonLibraryNvoxels'])
        self._min = np.array(self.detector['PhotonLibraryVolumeMin'])
        self._max = np.array(self.detector['PhotonLibraryVolumeMax'])
        self._vis = np.zeros((np.prod(self.nvoxels), self.nchannels))
        
    def convert(self, outfile):
        f = uproot.open(self.root_file + ":pmtresponse/PhotonLibraryData").arrays(library="np")
        self._vis[f['Voxel'], f['OpChannel']] = f['Visibility']
        hf = h5py.File(outfile, 'w')
        hf.create_dataset('max', data=self._max)
        hf.create_dataset('min', data=self._min)
        hf.create_dataset('numvox', data=self.nvoxels)
        hf.create_dataset('vis', data=self._vis, dtype="<f4")

if __name__ == "__main__":
    infile = "PhotonLibrary-20180801.root"
    outfile = "plib_20180801.h5"
    converter = Converter(root_file = infile)
    converter.convert(outfile)
    



        



        
        



