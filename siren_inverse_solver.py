import os, argparse
import shutil, utils
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from photon_library import PhotonLibrary
from flashmatch_manager import FlashMatchManager
from algorithms.match_modules import SirenFlash
from utils import DataWrapper, collate_fn
from algorithms.match_model import PoissonMatchLoss

class SirenInverseSolver():
    """
    Siren Inverse Solver using flash match data
    """
    def __init__(self, det_file, cfg_file, plib, particleana=None, opflashana=None):
        self.configure(det_file, cfg_file, plib, particleana, opflashana)

    def configure(self, det_file, cfg_file, plib, particleana, opflashana):
        self.detector_specs = yaml.load(open(det_file), Loader=yaml.Loader)['DetectorSpecs']
        config = yaml.load(open(cfg_file), Loader=yaml.Loader)['SirenInverse']

        self.num_tracks = int(config['NumTracks'])
        self.num_batches = int(config['NumBatches'])
        self.num_epochs = int(config['NumEpochs'])
        self.epochs_til_checkpoint = int(config['EpochsTilCheckpoint'])
        self.lr = config['LearningRate']

        self.mgr = FlashMatchManager(det_file, cfg_file, particleana, opflashana, plib)
        self.model = SirenFlash(self.mgr.flash_algo)
        self.optim = torch.optim.AdamW(lr=self.lr, params=self.model.parameters(), amsgrad=True)
        self.loss_fn = PoissonMatchLoss()

    def train(self, model_dir):
        epoch_start = 0
        total_steps = 0
        train_losses = []
        
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
            pass

        checkpoints_dir = os.path.join(model_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        print('Loading Data...')
        flashmatch_input = self.mgr.make_flashmatch_input(self.num_tracks * self.num_batches)
        train_data = DataWrapper(flashmatch_input.raw_qcluster_v, flashmatch_input.flash_v)
        dataloader = DataLoader(train_data, shuffle=True, batch_size=self.num_tracks, pin_memory=False, collate_fn=collate_fn)

        print("Training...")
        for epoch in range(epoch_start, self.num_epochs):
            for (track_v, flash_v) in dataloader:
                total_loss = 0
                for i in range(self.num_tracks):
                    track, gt_flash = track_v[i], flash_v[i]
                    pred_flash = self.model(track)

                    loss = self.loss_fn(pred_flash, gt_flash)
                    total_loss += loss
                total_loss /= self.num_tracks
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                train_losses.append(total_loss.item())
                total_steps += 1

            if not epoch % self.epochs_til_checkpoint and epoch:
                print('epoch:', epoch )

                torch.save({
                            'step': total_steps,
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optim.state_dict(),
                            'loss': train_losses,
                            },  os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                torch.save(self.model.model.state_dict(),
                        os.path.join(checkpoints_dir, 'siren_model_epoch_%04d.pth' % epoch))
                
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                        np.array(train_losses))

                plt_name = os.path.join(checkpoints_dir, 'total_loss_epoch_%04d.png' % epoch)
                utils.plot_losses(total_steps, train_losses, plt_name)
        
        torch.save(self.model.state_dict(),
               os.path.join(checkpoints_dir, 'model_final.pth'))
        torch.save(self.model.model.state_dict(),
               os.path.join(checkpoints_dir, 'siren_model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                np.array(train_losses))        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run Siren inverse solving')

    parser.add_argument('--cfg', default='data/flashmatch.cfg')
    parser.add_argument('--det', default='data/detector_specs.yml')
    parser.add_argument('--model_dir', '-m', default='models/experiment')
    args = parser.parse_args()

    cfg_file = args.cfg
    det_file = args.det
    model_dir = args.model_dir

    plib = PhotonLibrary()
    siren_solver = SirenInverseSolver(det_file, cfg_file, plib)
    siren_solver.train(model_dir)


