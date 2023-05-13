import torch
import numpy as np
import glob
import os
import getpy as gp
from src.utils import tk_mots


class MOTS_Dataset(torch.utils.data.Dataset):
    def __init__(self, windows, r=2, window_size=20):
        
        self.windows = windows
        self.r = r
        self.window_size = window_size
        self.mots_utils = tk_mots.MOTS_Utils(r)

    def __len__(self):
        return len(self.windows)


    def __getitem__(self, idx):
        MOTS, voxels = self.mots_utils.to_mts_fast(self.windows[idx], self.window_size)
        return MOTS, voxels


class MOTS_File_Dataset(torch.utils.data.Dataset):
    '''
        Takes a sparse tensor as an input and creates a dense MTS for each voxel in this sparse tensor.
    '''

    def __init__(self, datadir, radius=1, window_size=10):
        self.datadir = datadir
        self.files = glob.glob(os.path.join(datadir, "*/*"))
        #print("nfiles: ",self.files)
        self.radius = radius
        self.window_size = window_size
        self.mots_utils = tk_mots.MOTS_Utils(radius=radius)
        self.d = self.mots_utils.d

    def __len__(self):
        # get #crops and windows
        return len(self.files)

    def __getitem__(self, idx):
        # loads one window of a crop from some scene and transforms to MTS
        MOTS, _ = self.mots_utils.to_mts_fast(torch.load(self.files[idx]), self.window_size)
        return MOTS

    def test(dataset):
        for i in range(20):
            sample = dataset[i]
            # mts shape will be smaller because not all voxels are active at the end of a sliding window
            print(sample.shape)
