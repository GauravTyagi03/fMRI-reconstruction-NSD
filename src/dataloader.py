import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class NSDDataset(Dataset):
    def __init__(self, 
                 fmri_path,          # path to the filtered and flattened fMRI data
                 stimulus_path,      # path to nsd_stimuli.hdf5
                 trial_idx_path,     # path to sub01_trial_idx.npy
                 transform_image=None,
                 transform_fmri=None):
        """
        Args:
            fmri_path: path to the filtered and flattened fMRI data
            stimulus_path: path to nsd_stimuli.hdf5
            trial_idx_path: path to sub01_trial_idx.npy (maps all 30k trials to 73k image IDs)
            sessions: list or int specifying which sessions to load (1–40)
        """
        self.fmri_path = fmri_path
        self.stimulus_path = stimulus_path
        self.trial_idx = np.load(trial_idx_path)
        self.fmri_data = np.load(fmri_path)
        if self.trial_idx.shape[0] == 2 and self.trial_idx.shape[1] != 2:
            self.trial_idx = self.trial_idx.T  # ensure shape [N, 2]
        self.transform_image = transform_image or transforms.ToTensor()
        self.transform_fmri = transform_fmri
        self.num_trials = self.trial_idx.shape[0]
        # Lazy open HDF5 handles
        self.stim_file = None

    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        # return thr idx'th element of the trail_idx array
        fmri_idx, img_idx = self.trial_idx[idx]
        fmri_array = self.fmri_data[fmri_idx] 
        fmri = self.transform_fmri(fmri_array) if self.transform_fmri else torch.tensor(fmri_array)

        # open stimuli file lazily
        if self.stim_file is None:
            self.stim_file = h5py.File(self.stimulus_path, "r")

        img = self.stim_file["imgBrick"][img_idx]  # [425, 425, 3], uint8
        img = Image.fromarray(img)
        img = self.transform_image(img)

        # return structure matching paper’s expected tuple
        return {
            "voxel": fmri,
            "image": img,
            "coco": img_idx
        }

    def __del__(self):
        for f in self.fmri_files.values():
            try: f.close()
            except: pass
        if self.stim_file:
            try: self.stim_file.close()
            except: pass


def get_dataloaders(fmri_path,
                    stimulus_path,
                    train_idx_path,
                    val_idx_path,
                    batch_size=32,
                    transform_image=None,
                    transform_fmri=None):

    train_ds = NSDDataset(fmri_path, stimulus_path, train_idx_path, transform_image=transform_image, transform_fmri=transform_fmri)
    val_ds = NSDDataset(fmri_path, stimulus_path, val_idx_path, transform_image=transform_image, transform_fmri=transform_fmri)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dl, val_dl, len(train_ds), len(val_ds)