import torch.utils.data as data
import torchvision.transforms as transforms
from skimage import io
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn as nn

import random
import torch
import h5py
# import tables
import numpy as np
from pathlib import Path
from skimage import color

class LoadData(data.Dataset):
    def __init__(self, Moire_path, Moire_GT_path, Clear_path, Clear_Moire_path):

        self.Moire_path = Moire_path
        self.Moire_GT_path = Moire_GT_path

        self.Clear_path = Clear_path
        self.Clear_Moire_path = Clear_Moire_path

    def transform(self, Moire, Moire_GT, Clear, Clear_Moire, Clear_GRAY, Moire_GRAY):
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(Moire, output_size = (256, 256))
        Moire_crop = TF.crop(Moire, i, j, h, w)
        Moire_GT_crop = TF.crop(Moire_GT, i, j, h, w)

        Clear_crop = TF.crop(Clear, i, j, h, w)
        Clear_Moire_crop = TF.crop(Clear_Moire, i, j, h, w)

        Clear_GRAY_crop = TF.crop(Clear_GRAY, i, j, h, w)
        Moire_GRAY_crop = TF.crop(Moire_GRAY, i, j, h, w)

        # Gray channel historgram
        Clear_GRAY_crop = Clear_GRAY_crop.resize((64, 64))
        Clear_crop_gray_histogram = torch.as_tensor(Clear_GRAY_crop.histogram())
        Clear_crop_gray_histogram = Clear_crop_gray_histogram / 64

        Moire_GRAY_crop = Moire_GRAY_crop.resize((64, 64))
        Moire_crop_gray_histogram = torch.as_tensor(Moire_GRAY_crop.histogram())
        Moire_crop_gray_histogram = Moire_crop_gray_histogram / 64

        # Transform to tensor
        Moire_crop = TF.to_tensor(Moire_crop)
        Moire_GT_crop = TF.to_tensor(Moire_GT_crop)
        Clear_crop = TF.to_tensor(Clear_crop)
        Clear_Moire_crop = TF.to_tensor(Clear_Moire_crop)

        # Normalize
        # Moire_crop = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225))(Moire_crop)
        # Moire_GT_crop = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225))(Moire_GT_crop)
        # Clear_crop = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225))(Clear_crop)
        # Clear_Moire_crop = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225))(Clear_Moire_crop)
        # Clear_crop_gray_histogram = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.225, 0.225, 0.225))(Clear_crop_gray_histogram)

        return Moire_crop, Moire_GT_crop, Clear_crop, Clear_Moire_crop, Clear_crop_gray_histogram, Moire_crop_gray_histogram

    def __getitem__(self, index):
        Moire = Image.open(self.Moire_path[index])
        Moire_GT = Image.open(self.Moire_GT_path[index])

        Clear = Image.open(self.Clear_path[index])
        Clear_Moire = Image.open(self.Clear_Moire_path[index])

        Clear_GRAY = Clear.convert('L')
        Moire_GRAY = Moire.convert('L')

        x, y, z, w, o, p = self.transform(Moire, Moire_GT, Clear, Clear_Moire, Clear_GRAY, Moire_GRAY)

        return x, y, z, w, o, p

    def __len__(self):
        return len(self.Moire_path)
