import numpy as np
import yaml
from root_numpy import root2array
from constants import ParticleMass
from algorithms.lightpath import LightPath
from algorithms.flashalgo import FlashAlgo
from flashmatch_types import FlashMatchInput, QCluster, Flash

def convert(energy_v, pdg_code):
    mass = ParticleMass[pdg_code]
    return energy_v - mass

def filter_energy_deposit(energy_v, threshold=0.1):
    energy_deposits = energy_v[:-1] - energy_v[1:]
    idx = np.concatenate([energy_deposits > threshold, [False]], axis=0)
    full_idx = np.logical_or(idx, np.roll(idx, 1))
    return full_idx

class ROOTInput:
    def __init__(self, particleana, opflashana, plib, det_file, cfg_file=None):
        self.detector = yaml.load(open(det_file), Loader=yaml.Loader)['DetectorSpecs']
        self.qcluster_algo = LightPath(self.detector, cfg_file)
        self.flash_algo = FlashAlgo(plib, cfg_file)
        self.periodTPC = [-1000, 1000]
        self.tpc_tree_name = "largeant_particletree"
        self.pmt_tree_name = "opflash_flashtree"
        if not cfg_file is None:
            self.configure(cfg_file)

        self._particles = root2array(particleana, self.tpc_tree_name)
        self._opflash = root2array(opflashana, self.pmt_tree_name)
        # check data consistency
        self._entries_to_event = np.unique(self._particles['event']).astype(np.int32)

    def __len__(self):
        return len(self._entries_to_event)

    def configure(self,cfg_file):
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)["ROOTInput"]
        self.periodTPC = config['PeriodTPC']
        # Truncate TPC tracks (readout effect)
        self.truncate_tpc_readout = config['TruncateTPCReadout']
        # Truncate TPC tracks with active volume
        self.truncate_tpc_active = config['TruncateTPCActive']
        # Shift TPC tracks (for MCTrack to look realistic, needs true timings)
        self.shift_tpc = config['ShiftXForMC']

        # Time window to match MCFlash and MCTrack
        self.matching_window = config['MatchingWindow']
        # Whether to exclude flashes too close to each other
        self.exclude_reflashing = config['ExcludeReflashing']
        self.tpc_tree_name = config['TPCTreeName']
        self.pmt_tree_name = config['PMTTreeName']

        self.clustering = config['Clustering']
        self.clustering_threshold = config['ClusteringThreshold']
        self.clustering_time_window = config['ClusteringTimeWindow']
        self.matching_window_opflash = config['MatchingWindowOpflash']

        # Set seed if there is any specified
        if 'NumpySeed' in config:
            seed = config['NumpySeed']
            if seed < 0:
                import time
                seed = int(time.time())
            np.random.seed(seed)

    def event_id(self,entry_id):
        return self._entries_to_event[entry_id]

    def entry_id(self,event_id):
        return np.where(self._entries_to_event == event_id)[0][0]

    def get_entry(self,entry):
        event = self._entries_to_event[entry]
        return self.get_event(event)

    def get_event(self,event):
        return self._particles[self._particles['event'] == event], self._opflash[self._opflash['event'] == event]

    def make_qcluster(self, particles, select_pdg=[], exclude_pdg=[]):
        pid_v = []
        xyzs_v = []
        ts_v = []

        for pid, p in enumerate(particles):
            # Only use specified PDG code if pdg_code is available
            if not self.clustering or (len(select_pdg) and (int(p['pdg_code']) not in select_pdg)) or int(p['pdg_code']) in exclude_pdg:
                continue
            xyzs = np.column_stack([p['x_v'],p['y_v'],p['z_v']]).astype(np.float64)
            if xyzs.shape[0] < 2:
                continue
            merged = False
            if self.clustering:
                # Check if there alreay exists a cluster overlapping in time
                for i, xyzs2 in enumerate(xyzs_v):
                    t2_min = np.min(ts_v[i])
                    t2_max = np.max(ts_v[i])
                    t_min = np.min(p['time_v'])
                    t_max = np.max(p['time_v'])
                    d = -1
                    if (t2_min >= t_min and t2_min <= max(t_max, t_min + self.clustering_time_window)) or (t_min >= t2_min and t_min <= max(t2_max, t2_min + self.clustering_time_window)):
                        idx = filter_energy_deposit(convert(p['energy_v'], int(p['pdg_code'])))
                        xyzs_v[i].append(xyzs)
                        ts_v[i] = np.hstack([ts_v[i], p['time_v']])
                        pid_v[i] = min(pid_v[i], pid)  # FIXME do we want this?
                        merged = True
                        break
            if not merged:
                #print('New particle', p['pdg_code'], np.min(p['time_v']*1e-3))
                #print(xyzs[:5])
                idx = filter_energy_deposit(convert(p['energy_v'], int(p['pdg_code'])))
                #print(convert(p['energy_v'][:50], int(p['pdg_code'])))
                #print(convert(p['energy_v'], int(p['pdg_code']))[idx][:50])
                xyzs_v.append([xyzs])
                ts_v.append(p['time_v'])
                pid_v.append(pid)
        
        # Now looping over xyzs and making QClusters
        qcluster_v = []
        all_pts_v = []

        for i in range(len(xyzs_v)):
            qcluster = QCluster()
            all_pts = QCluster()
            #time = ts_v[i][0][0]
            for j in range(len(xyzs_v[i])):
                traj = xyzs_v[i][j]
                # If configured, truncate the physical boundary here
                # (before shifting or readout truncation)
                # if self._truncate_tpc_active:
                #     bbox = self.det.ActiveVolume()
                #     traj = self.geoalgo.BoxOverlap(bbox,traj)
                # Need at least 2 points to make QCluster
                if len(traj) < 2: continue;
                # Make QCluster
                qcluster += self.qcluster_algo.make_qcluster_from_track(traj)
                all_pts  += qcluster
                #time = min(time, np.min(ts_v[i][j]) * 1e-3)
            # If configured, truncate the physical boundary here
            if self.truncate_tpc_active:
                pt_min, pt_max = self.detector['ActiveVolumeMin'], self.detector['ActiveVolumeMax']
                qcluster.drop(pt_min[0],pt_max[0],pt_min[1],
                              pt_max[1],pt_min[2],pt_max[2])
            # need to be non-empty
            if len(qcluster) == 0: continue
            #ts=p['time_v']
            qcluster.time_true = np.min(ts_v[i]) * 1.e-3
            all_pts.time_true = qcluster.time_true
            #print('QCluster @ ', qcluster.time_true)
            # if qcluster.min_x() < -365:
            #     print("touching ", qcluster.min_x(), qcluster.time_true)
            #if qcluster.min_x() < self.det.ActiveVolume().Min()[0]:
            #    raise Exception('** wrong  *** ', qcluster.min_x(), qcluster.max_x(), qcluster.time_true)

            # Assign the index number of a particle
            qcluster.idx = pid_v[i]
            all_pts.idx = pid_v[i]
            qcluster_v.append(qcluster)
            all_pts_v.append(all_pts)
        return qcluster_v,all_pts_v

    def make_flash(self,opflash):
        flash_v = []
        for f_idx, f in enumerate(opflash):
            flash = Flash()
            flash.idx = f_idx
            pe_reco_v = f['pe_v']
            pe_true_v = f['pe_true_v']
            for pmt in range(self.detector['NOpDets']):
                flash.pe_v.append(pe_reco_v[pmt])
                flash.pe_err_v.append(0.)
                flash.time = f['time']
                flash.pe_true_v.append(pe_true_v[pmt])
                flash.time_true = f['time_true']
            if np.sum(flash.pe_v) > 0:
                flash.to_torch()
                flash_v.append(flash)
        return flash_v

    def make_flashmatch_input(self, entry):
        """
        Make sample from ROOT files
        """
        if entry > len(self):
            raise IndexError
        result = FlashMatchInput()
        # Find the list of sim::MCTrack entries for this event
        particles, opflash = self.get_entry(entry)
        # result.raw_particles = particles
        result.raw_qcluster_v, result.all_pts_v = self.make_qcluster(particles,select_pdg=[13],exclude_pdg=[2112, 1000010020, 1000010030, 1000020030, 1000020040])
        result.qcluster_v = [tpc.copy() for tpc in result.raw_qcluster_v]

         # If configured, shift X (for MCTrack to imitate reco)
        if self.shift_tpc:
            for i, qcluster in enumerate(result.qcluster_v):
                qcluster.xshift = qcluster.time_true * self.detector['DriftVelocity']
                if qcluster.xmin - self.detector['ActiveVolumeMin'][0] < self.detector['ActiveVolumeMax'][0] - qcluster.xmax:
                    result.qcluster_v[i] = qcluster.shift(qcluster.xshift) #+ (qcluster.min_x() - self.det.ActiveVolume().Min()[0])
                else:
                    result.qcluster_v[i] = qcluster.shift(-qcluster.xshift) #- (self.det.ActiveVolume().Max()[0] - qcluster.min_x())

        if self.truncate_tpc_readout:
            # Define allowed X recording regions
            min_tpcx, max_tpcx = [t * self.detector['DriftVelocity'] for t in self.periodTPC]
            for tpc in result.qcluster_v: tpc.drop(min_tpcx,max_tpcx)

        # Find the list of recob::OpFlash entries for this event
        result.flash_v = self.make_flash(opflash)

        # compute dt to previous and next flash
        for pmt in result.flash_v:
            pmt.dt_prev,pmt.dt_next = np.inf, np.inf
        for pmt in result.flash_v:
            for pmt2 in result.flash_v:
                if pmt2.idx == pmt.idx: continue
                if pmt2.time < pmt.time and abs(pmt2.time - pmt.time) < pmt.dt_prev:
                    pmt.dt_prev = abs(pmt2.time - pmt.time)
                if pmt2.time > pmt.time and abs(pmt.time - pmt2.time) < pmt.dt_next:
                    pmt.dt_next = abs(pmt.time - pmt2.time)
            #print(pmt.time, pmt.dt_prev, pmt.dt_next)

        # Exclude flashes too close apart
        if self.exclude_reflashing:
            selected = []
            for pmt in result.flash_v:
                if pmt.dt_prev > dt_threshold and pmt.dt_next > dt_threshold:
                    selected.append(pmt)
                else:
                    print('dropping', pmt.dt_prev, pmt.dt_next)
            result.flash_v = selected

        # Assign idx now based on true timings
        tpc_matched = []
        pmt_matched = []
        for pmt in result.flash_v:
            for tpc in result.qcluster_v:
                if tpc.idx in tpc_matched: continue
                dt = abs(pmt.time_true - tpc.time_true)
                if dt < self.matching_window:
                    result.true_match.append((pmt.idx, tpc.idx))
                    tpc_matched.append(tpc.idx)
                    pmt_matched.append(pmt.idx)
                    break
        # Assign idx based on opflash timings for tpc that have not been matched
        pmt_matched_second = []
        for tpc in result.qcluster_v:
            if tpc.idx in tpc_matched: continue
            possible_pmt_match = []
            for pmt in result.flash_v:
                if pmt.idx in pmt_matched or pmt.idx in pmt_matched_second: continue
                if pmt.time_true > 1e5: # this opflash has no mcflash
                    dt = (pmt.time - tpc.time_true)
                    if dt > self.matching_window_opflash[0] and dt < self.matching_window_opflash[1]:
                        possible_pmt_match.append((pmt.idx, dt))
            if len(possible_pmt_match) == 0: continue
            # If several opflashes can be matched, select the closest one
            # FIXME what if several tpc for same opflash?
            pmt_idx_min = possible_pmt_match[0][0]
            dt_min = possible_pmt_match[0][1]
            for pmt_idx, dt in possible_pmt_match:
                if dt < dt_min:
                    pmt_idx_min = pmt_idx
                    dt_min = dt
            result.true_match.append((pmt_idx_min, tpc.idx))
            pmt_matched_second.append(pmt_idx_min)
        return result