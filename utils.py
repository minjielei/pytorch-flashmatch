import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os, glob

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

class DataWrapper(DataLoader):
    def __init__(self, input, target):
        # self.dataset = dataset
        self.mgrid = input
        self.gt = target
        
    def __len__(self):
        return len(self.mgrid)
    
    def __getitem__(self, idx):
        in_dict = self.mgrid[idx].qpt_v
        gt_dict = self.gt[idx].pe_v

        return in_dict, gt_dict

def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    """
    This function will load the most current training checkpoint.
    """
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        total_steps = checkpoint['step']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losslogger = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {}, steps {})"
                  .format(filename, checkpoint['epoch'], checkpoint['step']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, total_steps, start_epoch, losslogger

def find_latest_checkpoint(model_dir):
    """
    This helper function finds the checkpoint with the largest
    epoch value given a model directory.
    """
    tmp_dir = os.path.join(model_dir, 'checkpoints')
    list_of_files = glob.glob(tmp_dir+'/model_epoch_*.pth') 
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def plot_losses(total_steps, train_losses, filename):
    x_steps = np.linspace(0, total_steps, num=total_steps)
    plt.figure(tight_layout=True)
    plt.plot(x_steps, train_losses)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()