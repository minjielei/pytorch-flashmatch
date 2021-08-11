import numpy as np
from flashmatch_manager import FlashMatchManager
from photon_library import PhotonLibrary
from utils import print_match_result

def demo(cfg_file, det_file, out_file='', num_tracks=None, num_entries=1):
    """
    Run function for ToyMC
    ---------
    Arguments
      cfg_file:   string for the location of a config file
      det_file:   string for the location of a detector spec file
      out_file:   string for an output analysis csv file path (optional)
      num_tracks: int for number of tracks to be generated each entry(optional)
      num_entries: number of entries(event) to run the matcher on
    """
    plib = PhotonLibrary()
    mgr = FlashMatchManager(plib, det_file, cfg_file)
    if num_tracks is not None:
        num_tracks = int(num_tracks)

    if out_file:
        import os
        if os.path.isfile(out_file):
            print('Output file',out_file,'already exists. Exiting...')
            return 

    np_result = None

    for entry in range(num_entries):
        match_input = mgr.make_flashmatch_input(num_tracks)
        match_v = mgr.match(match_input)

        if not out_file:
            print_match_result(match_input, match_v)
            continue
        
        all_matches = []
        for idx, (tpc_id, flash_id) in enumerate(zip(match_v.tpc_ids, match_v.flash_ids)):
            qcluster, flash = match_input.qcluster_v[tpc_id], match_input.flash_v[flash_id]
            raw_qcluster = match_input.raw_qcluster_v[tpc_id]
            loss, reco_dx, reco_pe = match_v.loss_v[idx], match_v.reco_x_v[idx], match_v.reco_pe_v[idx]
            matched = (tpc_id, flash_id) in match_input.true_match
            store = np.array([[
                entry,
                loss,
                qcluster.idx,
                flash.idx,
                raw_qcluster.xmin,
                raw_qcluster.xmax,
                qcluster.xmin,
                qcluster.xmax,
                qcluster.xmin + reco_dx,
                qcluster.xmax + reco_dx,
                int(matched),
                len(qcluster),
                qcluster.sum(),
                qcluster.length(),
                qcluster.true_time,
                reco_pe,
                flash.sum(),
                flash.time,
                flash.true_time
            ]])
            all_matches.append(store)
        if out_file and len(all_matches):
            np_event = np.concatenate(all_matches, axis=0)
            if np_result is None:
                np_result = np_event
            else:
                np_result = np.concatenate([np_result,np_event],axis=0)

    if not out_file:
        return

    np.savetxt(out_file, np_result, delimiter=',', header=','.join(attribute_names()))

def attribute_names():

    return [
        'entry',
        'loss',
        'tpc_idx',
        'flash_idx',
        'true_min_x',
        'true_max_x',
        'qcluster_min_x',
        'qcluster_max_x',
        'reco_min_x',
        'reco_max_x',
        'matched',
        'qcluster_num_points',
        'qcluster_sum',
        'qcluster_length',
        'qcluster_time_true',
        'hypothesis_sum',  # Hypothesis flash sum
        'flash_sum', # OpFlash Sum
        'flash_time',
        'flash_time_true',
    ]