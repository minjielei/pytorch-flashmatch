import os, argparse
import shutil, utils
import yaml
import torch
import numpy as np
from functools import partial
from torch.utils.data import DataLoader
from photon_library import PhotonLibrary
from toymc_siren import ToyMCSiren
from algorithms.match_modules import SirenFlash
from utils import DataWrapper, collate_fn

class SirenInverseSolver():
    """
    Siren Inverse Solver using flash match data
    """
    def __init__(self, det_file, cfg_file, particleana=None, opflashana=None):
        self.configure(det_file, cfg_file, particleana, opflashana)

    def configure(self, det_file, cfg_file, particleana, opflashana):
        self.detector_specs = yaml.load(open(det_file), Loader=yaml.Loader)['DetectorSpecs']
        self.cfg_file = cfg_file
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)['SirenInverse']

        self.model_dir = config['ModelDir']
        self.experiment_name = config['ExperimentName']
        self.plib_file = config['PlibFile']
        self.num_tracks = int(config['NumTracks'])
        self.num_batches = int(config['NumBatches'])
        self.num_epochs = int(config['NumEpochs'])
        self.steps_til_summary = int(config['StepsTilSummary'])
        self.epochs_til_checkpoint = int(config['EpochsTilCheckpoint'])
        self.lr = config['LearningRate']

        self.plib = PhotonLibrary(self.plib_file)

        self.mgr = ToyMCSiren(self.plib, det_file, cfg_file)
        self.model = SirenFlash(self.mgr.flash_algo)
        self.optim = torch.optim.Adam(lr=self.lr, params=self.model.parameters())
        self.loss_fn = PoissonMatchLoss()

    def train(self):
        epoch_start = 0
        total_steps = 0
        train_losses = []

        model_dir = os.path.join(self.model_dir, self.experiment_name)
        
        if os.path.exists(model_dir+'/checkpoints'):
            val = input("The model directory %s exists. Load latest run? (y/n)" % model_dir)
            if val == 'y':
                filename = utils.find_latest_checkpoint(model_dir)
                self.model, self.optim, total_steps, epoch_start, train_losses =  utils.load_checkpoint(self.model, self.optim, filename)
                print(self.optim.param_groups[0]['lr'])
                if val == 'n':
                    shutil.rmtree(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        shutil.copy(self.cfg_file, model_dir)

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        print('Loading Data...')
        flashmatch_input = self.mgr.make_flashmatch_input(self.num_tracks * self.num_batches)
        train_data = DataWrapper(flashmatch_input.raw_qcluster_v, flashmatch_input.flash_v)
        dataloader = DataLoader(train_data, shuffle=True, batch_size=self.num_tracks, pin_memory=False, collate_fn=collate_fn)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='min', factor=0.5, patience=50, threshold=1e-4, 
            threshold_mode='rel', cooldown=10, verbose=True)

        print("Training...")
        for epoch in range(epoch_start, self.num_epochs):
            for (track_v, flash_v) in dataloader:
                total_loss = 0
                for i in range(len(track_v)):
                    track, gt_flash = track_v[i], flash_v[i] 
                    weight = self.plib.WeightFromPos(track[:, :3])
                    pred_flash = self.model(track)

                    loss = self.loss_fn(pred_flash, gt_flash, weight)
                    total_loss += loss
                total_loss /= len(track_v)
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                train_losses.append(total_loss.item())
                total_steps += 1

                if not total_steps % self.steps_til_summary:
                    torch.save({
                        'step': total_steps,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'loss': train_losses,
                        },  os.path.join(checkpoints_dir, 'model_current.pth'))
                    
            scheduler.step(total_loss)
            
            if not epoch % self.epochs_til_checkpoint and epoch:
                print('epoch:', epoch )

                torch.save({
                        'step': total_steps,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'loss': train_losses,
                        },  os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                torch.save({'model_state_dict': self.model.model.state_dict()},
                        os.path.join(checkpoints_dir, 'siren_model_current.pth'))
                
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_current.txt'),
                        np.array(train_losses))

                plt_name = os.path.join(checkpoints_dir, 'total_loss_current.png')
                utils.plot_losses(total_steps, train_losses, plt_name)
        
        torch.save({'model_state_dict': self.model.state_dict()},
               os.path.join(checkpoints_dir, 'model_final.pth'))
        torch.save({'model_state_dict': self.model.model.state_dict()},
               os.path.join(checkpoints_dir, 'siren_model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                np.array(train_losses))
        
        #Plot and save loss
        plt_name = os.path.join(checkpoints_dir, 'total_loss_final.png')
        utils.plot_losses(total_steps, train_losses, plt_name)

def log_mse_loss(pred, gt, weight=1.):
    H = torch.clamp(pred, min=1.)
    O = torch.clamp(gt, min=1.)
    return (weight * (torch.log10(O) - torch.log10(H)) ** 2).mean()

class PoissonMatchLoss(torch.nn.Module):
    """
    Poisson NLL Loss for gradient-based optimization model
    """
    def __init__(self):
        super(PoissonMatchLoss, self).__init__()
        self.poisson_nll = torch.nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")

    def forward(self, input, target, weight=1.):
        H = torch.clamp(input, min=0.01)
        O = torch.clamp(target, min=0.01)
        loss = self.poisson_nll(H, O)
        return torch.mean(weight * loss)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run Siren inverse solving')

    parser.add_argument('--cfg', default='data/flashmatch.cfg')
    parser.add_argument('--det', default='data/detector_specs.yml')
    args = parser.parse_args()

    cfg_file = args.cfg
    det_file = args.det

    siren_solver = SirenInverseSolver(det_file, cfg_file)
    siren_solver.train()