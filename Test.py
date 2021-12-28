import argparse
import os

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
import glob
import numpy as np
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid

import itertools
from math import log10
from PIL import Image
from skimage import io
from skimage.transform import resize
from torch.autograd import Variable
from Models import *
from Loss_func_demoire import L1_ASL, Sobel_Grads
from utils import *
import kornia
import time

from torch.autograd import Variable

from loadData import LoadData
import torch.autograd as autograd
import torch.utils.tensorboard as tfboard
from IQA_pytorch import SSIM, utils

parser = argparse.ArgumentParser(description = 'TRAINING DEMOIERING MODEL')
parser.add_argument("--epoch", type=int, default=50, help="epoch to start training from")
parser.add_argument('--num_epochs', default = 150, type = int)
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument("--save_data_name", type=str, default="CML", help="name of the dataset")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--is_cuda', default='cuda:0', type = str)


def image2tensor(img):
    out = img.swapaxes(0, 2).swapaxes(1, 2)
    out = torch.from_numpy(out * 1.0)
    out = out / 255.0

    channel, height, width = out.size()
    out = torch.reshape(out, (1, channel, height, width))

    return out

if __name__ == '__main__':
    opt = parser.parse_args()

    epoch = opt.epoch
    NUM_EPOCHS = opt.num_epochs
    BATCH = opt.batch_size
    cuda = opt.is_cuda

    Moire_path = 'DATA/Dataset/train/Unpaired_Moire/*.*'
    Moire_GT_path = 'DATA/Dataset/train/Unpaired_Moire_GT/*.*'
    Clear_path = 'DATA/Dataset/train/Unpaired_Clear/*.*'
    Clear_Moire_path = 'DATA/Dataset/train/Unpaired_Clear_Moire/*.*'

    # Create data train
    MOIRE = sorted(glob.glob(Moire_path))
    MOIRE_GT = sorted(glob.glob(Moire_GT_path))
    CLEAR = sorted(glob.glob(Clear_path))
    CLEAR_MOIRE = sorted(glob.glob(Clear_Moire_path))

    moire_dataset = LoadData(Moire_path=MOIRE, Moire_GT_path=MOIRE_GT, Clear_path=CLEAR, Clear_Moire_path=CLEAR_MOIRE)

    data_train_loader = torch.utils.data.DataLoader(moire_dataset, batch_size=BATCH, shuffle=True, num_workers=4)

    # TensorBoard
    tf_board_logs = 'TensorBoard/'
    time_before = datetime.datetime.now()
    tfb_train_writer = tfboard.SummaryWriter(log_dir=os.path.join(tf_board_logs, 'train_{}'.format(time_before)))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Custom weights initialization called on network
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    # LOSS FUNCTIONS
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_MSE = torch.nn.MSELoss()
    criterion_content = L1_ASL()
    Loss = L1_ASL()

    Net_Demoire_256 = GM2_UNet5_256(6, 3)
    Net_Demoire_128 = GM2_UNet5_128(6, 3)
    Net_Demoire_64 = GM2_UNet5_64(3, 3)
    Net_Demoire_TMB = TMB(256, 1)

    if cuda:
        Net_Demoire_256 = Net_Demoire_256.to(cuda)
        Net_Demoire_128 = Net_Demoire_128.to(cuda)
        Net_Demoire_64 = Net_Demoire_64.to(cuda)
        Net_Demoire_TMB = Net_Demoire_TMB.to(cuda)

        # LOSS FUNCTIONS
        criterion_cycle.to(cuda)
        criterion_GAN.to(cuda)
        criterion_content.to(cuda)
        criterion_MSE.to(cuda)

    if opt.epoch != 0:
        # Load pretrained models
        Net_Demoire_256.load_state_dict(
            torch.load("MODEL/Pretrained/Net_Demoire_256/Net_Demoire_256_%d.pth" % (opt.epoch)))
        Net_Demoire_128.load_state_dict(
            torch.load("MODEL/Pretrained/Net_Demoire_128/Net_Demoire_128_%d.pth" % (opt.epoch)))
        Net_Demoire_64.load_state_dict(
            torch.load("MODEL/Pretrained/Net_Demoire_64/Net_Demoire_64_%d.pth" % (opt.epoch)))
        Net_Demoire_TMB.load_state_dict(
            torch.load("MODEL/Pretrained/Net_Demoire_TMB/Net_Demoire_TMB_%d.pth" % (opt.epoch)))


    # Saving to plot for training
    L1_losses = []
    downx2 = nn.UpsamplingNearest2d(scale_factor=0.5)
    upx2 = nn.UpsamplingNearest2d(scale_factor=2)

    for epoch in range(opt.epoch, NUM_EPOCHS):
        iteration = 0

        # Validation
        in_path = sorted(glob.glob('DATA/Dataset/Unpaired_test/Unpaired_Moire/*.*'))
        gt_path = sorted(glob.glob('DATA/Dataset/Unpaired_test/Unpaired_Moire_GT/*.*'))

        avg_psnr = 0

        with torch.no_grad():
            Net_Demoire_256.eval()
            Net_Demoire_128.eval()
            Net_Demoire_64.eval()
            Net_Demoire_TMB.eval()

            for i in range(len(in_path)):
                input_ = io.imread(in_path[i])
                input_ = image2tensor(input_)
                input_ = input_.to(cuda).float()

                gt_ = io.imread(gt_path[i])
                gt_ = image2tensor(gt_)
                gt_ = gt_.to(cuda).float()

                input_downx2_ = downx2(input_)
                input_downx4_ = downx2(downx2(input_))

                # forward
                out_downx4_ = Net_Demoire_64(input_downx4_)
                temp_ = out_downx4_

                out_downx4_ = torch.squeeze(downx2(downx2(out_downx4_)))

                trans = torchvision.transforms.ToPILImage()

                out_downx4_ = trans(out_downx4_)

                # histogram
                out_downx4_gray = transforms.Grayscale(num_output_channels=1)(out_downx4_)
                out_downx4_gray_tensor_ = torch.as_tensor(out_downx4_gray.histogram())

                out_downx4_gray_tensor_ = torch.unsqueeze(out_downx4_gray_tensor_, 0)

                out_downx4_gray_tensor_ = out_downx4_gray_tensor_ / 64
                out_downx4_gray_tensor_ = out_downx4_gray_tensor_.to(cuda).float()

                alpha_ = Net_Demoire_TMB(out_downx4_gray_tensor_)
                alpha_ = torch.as_tensor(alpha_)
                alpha_ = alpha_ + (1e-3)

                final_out_downx4_ = temp_ / alpha_
                gradual_fusion_x2_x4_ = torch.cat((input_downx2_.detach(), upx2(final_out_downx4_.detach())), dim=1)
                out_downx2_ = Net_Demoire_128(gradual_fusion_x2_x4_)
                final_out_downx2_ = out_downx2_ / alpha_

                gradual_fusion_x_x2_ = torch.cat((input_.detach(), upx2(final_out_downx2_.detach())), dim=1)

                out_ = Net_Demoire_256(gradual_fusion_x_x2_)
                final_out_ = out_ / alpha_

                # Calculate PSNR
                mse = criterion_MSE(final_out_, gt_)
                psnr = 10 * log10(1.0 / mse.item())
                avg_psnr += psnr

                # Calculate SSIM
                model = SSIM(channels=3).cuda()
                score = model(final_out_, gt_, as_loss=False)
                score = score.item()

                # Arange images along x-axis
                final_out_ = make_grid(final_out_, nrow=5, normalize=True)
                input_ = make_grid(input_, nrow=5, normalize=True)
                GT_ = make_grid(gt_, nrow=5, normalize=True)

                image_grid = final_out_
                save_image(image_grid, "MODEL/%s/IMAGE/%s/%s.png" % (opt.save_data_name, epoch, i), normalize=False)

            print("VALIDATION PSNR: ===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(in_path)))
            print("VALIDATION SSIM: ===> Avg. SSIM: {:.4f} dB".format(score))
            break

