import numpy as np

# helper function to print out match results
def print_match_result(flashmatch_input, match):
    qcluster_v, flash_v = flashmatch_input.qcluster_v, flashmatch_input.flash_v
    for idx, (i, j) in enumerate(zip(match.tpc_ids, match.flash_ids)):
        print('Match ID: ', idx)
        loss, reco_x, reco_pe = match.loss_v[idx], match.reco_x_v[idx], match.reco_pe_v[idx]
        reco_x += qcluster_v[i].xmin
        track_id, flash_id = qcluster_v[i].idx, flash_v[j].idx
        correct = (flash_id, track_id) in flashmatch_input.true_match
        true_x = flashmatch_input.raw_qcluster_v[i].xmin
        true_pe = flash_v[j].sum()
        template = """PMT/TPC IDs {}/{}, Loss {:.5f}, Correct? {}, reco vs. true: X {:.5f} vs. {:.5f}, PE {:.5f} vs. {:.5f}"""
        print(template.format(flash_id, track_id, loss, correct, reco_x, true_x, reco_pe, true_pe))