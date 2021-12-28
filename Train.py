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

from torch.autograd import Variable

from loadData import LoadData
import torch.autograd as autograd
import torch.utils.tensorboard as tfboard
from IQA_pytorch import SSIM, utils

parser = argparse.ArgumentParser(description = 'TRAINING DEMOIERING MODEL')
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
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

    # Initialize generator and discriminator
    Net_Demoire_256 = GM2_UNet5_256(6, 3)
    Net_Demoire_128 = GM2_UNet5_128(6, 3)
    Net_Demoire_64 = GM2_UNet5_64(3, 3)
    Net_Demoire_TMB = TMB(256, 1)

    # 256 size
    G_Artifact_256_2 = GM2_UNet5_256(6, 3)
    D_Moire_256 = Discriminator(6, 256, 256)
    D_Clear_256 = Discriminator(6, 256, 256)

    # 128 size
    G_Artifact_128_2 = GM2_UNet5_128(6, 3)
    D_Moire_128 = Discriminator(6, 128, 128)
    D_Clear_128 = Discriminator(6, 128, 128)

    # 64 size
    G_Artifact_64_1 = TMB(256, 1)
    G_Artifact_64_2 = GM2_UNet5_64(6, 3)
    D_Moire_64 = Discriminator(6, 64, 64)
    D_Clear_64 = Discriminator(6, 64, 64)

    # LOSS FUNCTIONS
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_MSE = torch.nn.MSELoss()
    criterion_content = L1_ASL()

    Loss = L1_ASL()

    if cuda:
        Net_Demoire_256 = Net_Demoire_256.to(cuda)
        Net_Demoire_128 = Net_Demoire_128.to(cuda)
        Net_Demoire_64 = Net_Demoire_64.to(cuda)
        Net_Demoire_TMB = Net_Demoire_TMB.to(cuda)

        # 256 size
        G_Artifact_256_2 = G_Artifact_256_2.to(cuda)
        D_Moire_256 = D_Moire_256.to(cuda)
        D_Clear_256 = D_Clear_256.to(cuda)

        # 128 size
        G_Artifact_128_2 = G_Artifact_128_2.to(cuda)
        D_Moire_128 = D_Moire_128.to(cuda)
        D_Clear_128 = D_Clear_128.to(cuda)

        # 64 size
        G_Artifact_64_1 = G_Artifact_64_1.to(cuda)
        G_Artifact_64_2 = G_Artifact_64_2.to(cuda)
        D_Moire_64 = D_Moire_64.to(cuda)
        D_Clear_64 = D_Clear_64.to(cuda)

        # LOSS FUNCTIONS
        criterion_cycle.to(cuda)
        criterion_GAN.to(cuda)
        criterion_content.to(cuda)
        criterion_MSE.to(cuda)

    if opt.epoch != 0:
        # Load pretrained models
        Net_Demoire_256.load_state_dict(
            torch.load("MODEL/%s/Net_Demoire_256/Net_Demoire_256_%d.pth" % (opt.save_data_name, opt.epoch)))
        Net_Demoire_128.load_state_dict(
            torch.load("MODEL/%s/Net_Demoire_128/Net_Demoire_128_%d.pth" % (opt.save_data_name, opt.epoch)))
        Net_Demoire_64.load_state_dict(
            torch.load("MODEL/%s/Net_Demoire_64/Net_Demoire_64_%d.pth" % (opt.save_data_name, opt.epoch)))
        Net_Demoire_TMB.load_state_dict(
            torch.load("MODEL/%s/Net_Demoire_TMB/Net_Demoire_TMB_%d.pth" % (opt.save_data_name, opt.epoch)))

        # 256 size
        G_Artifact_256_2.load_state_dict(
            torch.load("MODEL/%s/G_Artifact_256_2/G_Artifact_256_2_%d.pth" % (opt.save_data_name, opt.epoch)))
        D_Moire_256.load_state_dict(
            torch.load("MODEL/%s/D_Moire_256/D_Moire_256_%d.pth" % (opt.save_data_name, opt.epoch)))
        D_Clear_256.load_state_dict(
            torch.load("MODEL/%s/D_Clear_256/D_Clear_256_%d.pth" % (opt.save_data_name, opt.epoch)))

        # 128 size
        G_Artifact_128_2.load_state_dict(
            torch.load("MODEL/%s/G_Artifact_128_2/G_Artifact_128_2_%d.pth" % (opt.save_data_name, opt.epoch)))
        D_Moire_128.load_state_dict(
            torch.load("MODEL/%s/D_Moire_128/D_Moire_128_%d.pth" % (opt.save_data_name, opt.epoch)))
        D_Clear_128.load_state_dict(
            torch.load("MODEL/%s/D_Clear_128/D_Clear_128_%d.pth" % (opt.save_data_name, opt.epoch)))

        # 64 size
        G_Artifact_64_1.load_state_dict(
            torch.load("MODEL/%s/G_Artifact_64_1/G_Artifact_64_1_%d.pth" % (opt.save_data_name, opt.epoch)))
        G_Artifact_64_2.load_state_dict(
            torch.load("MODEL/%s/G_Artifact_64_2/G_Artifact_64_2_%d.pth" % (opt.save_data_name, opt.epoch)))
        D_Moire_64.load_state_dict(
            torch.load("MODEL/%s/D_Moire_64/D_Moire_64_%d.pth" % (opt.save_data_name, opt.epoch)))
        D_Clear_64.load_state_dict(
            torch.load("MODEL/%s/D_Clear_64/D_Clear_64_%d.pth" % (opt.save_data_name, opt.epoch)))


    else:
        # Initialize weights
        Net_Demoire_256.apply(weights_init)
        Net_Demoire_128.apply(weights_init)
        Net_Demoire_64.apply(weights_init)
        Net_Demoire_TMB.apply(weights_init)

        # 256 size
        G_Artifact_256_2.apply(weights_init)
        D_Moire_256.apply(weights_init)
        D_Clear_256.apply(weights_init)

        # 128 size
        G_Artifact_128_2.apply(weights_init)
        D_Moire_128.apply(weights_init)
        D_Clear_128.apply(weights_init)

        # 64 size
        G_Artifact_64_1.apply(weights_init)
        G_Artifact_64_2.apply(weights_init)
        D_Moire_64.apply(weights_init)
        D_Clear_64.apply(weights_init)


    # Optimizers
    optimizer_Net_Demoire_256 = torch.optim.AdamW(Net_Demoire_256.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Net_Demoire_128 = torch.optim.AdamW(Net_Demoire_128.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Net_Demoire_64 = torch.optim.AdamW(Net_Demoire_64.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_Net_Demoire_TMB = torch.optim.AdamW(Net_Demoire_TMB.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 256 size
    optimizer_G_256_2 = torch.optim.AdamW(G_Artifact_256_2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_Moire_256 = torch.optim.AdamW(D_Moire_256.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_Clear_256 = torch.optim.AdamW(D_Clear_256.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 128 size
    optimizer_G_128_2 = torch.optim.AdamW(G_Artifact_128_2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_Moire_128 = torch.optim.AdamW(D_Moire_128.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_Clear_128 = torch.optim.AdamW(D_Clear_128.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 64 size
    optimizer_G_64_1 = torch.optim.AdamW(G_Artifact_64_1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G_64_2 = torch.optim.AdamW(G_Artifact_64_2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_Moire_64 = torch.optim.AdamW(D_Moire_64.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D_Clear_64 = torch.optim.AdamW(D_Clear_64.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Step LR
    lr_scheduler_Net_256 = optim.lr_scheduler.StepLR(optimizer_Net_Demoire_256, step_size = 100, gamma = 0.1)
    lr_scheduler_Net_128 = optim.lr_scheduler.StepLR(optimizer_Net_Demoire_128, step_size = 100, gamma = 0.1)
    lr_scheduler_Net_64 = optim.lr_scheduler.StepLR(optimizer_Net_Demoire_64, step_size = 100, gamma = 0.1)
    lr_scheduler_Net_Demoire_TMB = optim.lr_scheduler.StepLR(optimizer_Net_Demoire_TMB, step_size = 100, gamma = 0.1)

    # 256
    lr_scheduler_G_256_2 = optim.lr_scheduler.StepLR(optimizer_G_256_2, step_size = 100, gamma = 0.1)
    lr_scheduler_D_Moire_256 = optim.lr_scheduler.StepLR(optimizer_D_Moire_256, step_size = 100, gamma = 0.1)
    lr_scheduler_D_Clear_256 = optim.lr_scheduler.StepLR(optimizer_D_Clear_256, step_size = 100, gamma = 0.1)

    # 128
    lr_scheduler_G_128_2 = optim.lr_scheduler.StepLR(optimizer_G_128_2, step_size = 100, gamma = 0.1)
    lr_scheduler_D_Moire_128 = optim.lr_scheduler.StepLR(optimizer_D_Moire_128, step_size = 100, gamma = 0.1)
    lr_scheduler_D_Clear_128 = optim.lr_scheduler.StepLR(optimizer_D_Clear_128, step_size = 100, gamma = 0.1)

    # 64
    lr_scheduler_G_64_1 = optim.lr_scheduler.StepLR(optimizer_G_64_1, step_size = 100, gamma = 0.1)
    lr_scheduler_G_64_2 = optim.lr_scheduler.StepLR(optimizer_G_64_2, step_size = 100, gamma = 0.1)
    lr_scheduler_D_Moire_64 = optim.lr_scheduler.StepLR(optimizer_D_Moire_64, step_size = 100, gamma = 0.1)
    lr_scheduler_D_Clear_64 = optim.lr_scheduler.StepLR(optimizer_D_Clear_64, step_size = 100, gamma = 0.1)

    # Saving to plot for training
    L1_losses = []

    downx2 = nn.UpsamplingNearest2d(scale_factor = 0.5)
    upx2 = nn.UpsamplingNearest2d(scale_factor = 2)

    for epoch in range(opt.epoch, NUM_EPOCHS):
        iteration = 0

        for i, data in enumerate(data_train_loader, 0):

            # load data
            MOIRE_256 = data[0].to(cuda)
            MOIRE_128 = downx2(MOIRE_256)
            MOIRE_64 = downx2(MOIRE_128)

            CLEAR_256 = data[2].to(cuda)
            CLEAR_128 = downx2(CLEAR_256)
            CLEAR_64 = downx2(CLEAR_128)

            historgram = data[4].to(cuda).float()
            histogram2 = data[5].to(cuda).float()

            valid_256 = Variable(Tensor(MOIRE_256.size(0), 1, 1, 1).fill_(1.0).to(cuda), requires_grad=False)
            fake_256 = Variable(Tensor(MOIRE_256.size(0), 1, 1, 1).fill_(0.0).to(cuda), requires_grad=False)

            valid_128 = Variable(Tensor(MOIRE_128.size(0), 1, 1, 1).fill_(1.0).to(cuda), requires_grad=False)
            fake_128 = Variable(Tensor(MOIRE_128.size(0), 1, 1, 1).fill_(0.0).to(cuda), requires_grad=False)

            valid_64 = Variable(Tensor(MOIRE_64.size(0), 1, 1, 1).fill_(1.0).to(cuda), requires_grad=False)
            fake_64 = Variable(Tensor(MOIRE_64.size(0), 1, 1, 1).fill_(0.0).to(cuda), requires_grad=False)


            # ------------------
            #  Train Generators 64 size
            # ------------------
            optimizer_G_64_1.zero_grad()
            optimizer_G_64_2.zero_grad()
            G_Artifact_64_1.train()
            G_Artifact_64_2.train()
            Net_Demoire_64.eval()
            Net_Demoire_TMB.eval()

            # GENERATOR-1
            learned_Moire_64 = G_Artifact_64_1(historgram)
            pseudo_Moire_64 = CLEAR_64 * learned_Moire_64

            pseudo_Moire_filter_64 = pseudo_Moire_64 - kornia.filters.median_blur(pseudo_Moire_64, (3, 3))
            pseudo_Moire_cat_64 = torch.cat((pseudo_Moire_64, pseudo_Moire_filter_64), dim=1)

            ## G1-GAN LOSS ##
            loss_GAN_64_1 = criterion_GAN(D_Moire_64(pseudo_Moire_cat_64), valid_64)

            # G2
            z_64_2 = Variable(Tensor(np.random.uniform(-1, 1, size=CLEAR_64.shape).astype(np.float32)).to(cuda))
            deep_real_Clean_Noise_64 = torch.cat((pseudo_Moire_64.detach(), z_64_2), dim=1)
            learned_Moire_64_2 = G_Artifact_64_2(deep_real_Clean_Noise_64)
            deep_pseudo_Moire_64 = pseudo_Moire_64.detach() + learned_Moire_64_2

            deep_pseudo_Moire_filter_64 = deep_pseudo_Moire_64 - kornia.filters.median_blur(deep_pseudo_Moire_64, (3, 3))
            deep_pseudo_Moire_cat_64 = torch.cat((deep_pseudo_Moire_64, deep_pseudo_Moire_filter_64), dim=1)

            ## G2-GAN LOSS 2 ##
            loss_GAN_64_2 = criterion_GAN(D_Moire_64(deep_pseudo_Moire_cat_64), valid_64)

            # G2-CYCLE LOSS
            # Real Moire(256 size) -> DemoireNet -> Demoire image -> Downsampling for Multi-GAN
            Demoire_64 = Net_Demoire_64(MOIRE_64).detach()

            # DEMOIRE HISTOGRAM
            Demoire_1, Demoire_2, Demoire_3, Demoire_4 = torch.chunk(Demoire_64, 4, dim=0)

            Demoire_1 = torch.squeeze(Demoire_1)
            Demoire_2 = torch.squeeze(Demoire_2)
            Demoire_3 = torch.squeeze(Demoire_3)
            Demoire_4 = torch.squeeze(Demoire_4)

            trans = torchvision.transforms.ToPILImage()

            Demoire_1 = trans(Demoire_1)
            Demoire_2 = trans(Demoire_2)
            Demoire_3 = trans(Demoire_3)
            Demoire_4 = trans(Demoire_4)

            Demoire_GRAY_1 = transforms.Grayscale(num_output_channels=1)(Demoire_1)
            Demoire_GRAY_2 = transforms.Grayscale(num_output_channels=1)(Demoire_2)
            Demoire_GRAY_3 = transforms.Grayscale(num_output_channels=1)(Demoire_3)
            Demoire_GRAY_4 = transforms.Grayscale(num_output_channels=1)(Demoire_4)

            Demoire_gray_tensor_1 = torch.as_tensor(Demoire_GRAY_1.histogram())
            Demoire_gray_tensor_2 = torch.as_tensor(Demoire_GRAY_2.histogram())
            Demoire_gray_tensor_3 = torch.as_tensor(Demoire_GRAY_3.histogram())
            Demoire_gray_tensor_4 = torch.as_tensor(Demoire_GRAY_4.histogram())

            Demoire_histogram = torch.stack((Demoire_gray_tensor_1, Demoire_gray_tensor_2, Demoire_gray_tensor_3, Demoire_gray_tensor_4), dim=0)

            Demoire_histogram = Demoire_histogram / 64
            Demoire_histogram = Demoire_histogram.to(cuda)

            Demoire_bright_alpha = Net_Demoire_TMB(Demoire_histogram).detach()
            Demoire_bright_alpha = Demoire_bright_alpha + (1e-3)

            final_Demoire_64 = Demoire_64 / Demoire_bright_alpha


            ######################################
            # final_Demoire_64 histogram
            ######################################

            # DEMOIRE HISTOGRAM
            D_1, D_2, D_3, D_4 = torch.chunk(final_Demoire_64, 4, dim=0)

            D_1 = torch.squeeze(D_1)
            D_2 = torch.squeeze(D_2)
            D_3 = torch.squeeze(D_3)
            D_4 = torch.squeeze(D_4)

            trans = torchvision.transforms.ToPILImage()

            D_1 = trans(D_1)
            D_2 = trans(D_2)
            D_3 = trans(D_3)
            D_4 = trans(D_4)

            D_GRAY_1 = transforms.Grayscale(num_output_channels=1)(D_1)
            D_GRAY_2 = transforms.Grayscale(num_output_channels=1)(D_2)
            D_GRAY_3 = transforms.Grayscale(num_output_channels=1)(D_3)
            D_GRAY_4 = transforms.Grayscale(num_output_channels=1)(D_4)

            D_gray_tensor_1 = torch.as_tensor(D_GRAY_1.histogram())
            D_gray_tensor_2 = torch.as_tensor(D_GRAY_2.histogram())
            D_gray_tensor_3 = torch.as_tensor(D_GRAY_3.histogram())
            D_gray_tensor_4 = torch.as_tensor(D_GRAY_4.histogram())

            final_Demoire_64_histogram = torch.stack(
                (D_gray_tensor_1, D_gray_tensor_2, D_gray_tensor_3, D_gray_tensor_4), dim=0)

            final_Demoire_64_histogram = final_Demoire_64_histogram / 64
            final_Demoire_64_histogram = final_Demoire_64_histogram.to(cuda)

            learned_Moire_2 = G_Artifact_64_1(final_Demoire_64_histogram).detach()

            Reconv_Moire_64_1 = final_Demoire_64 * learned_Moire_2

            deep_Demoire_Noise_64 = torch.cat((Reconv_Moire_64_1, z_64_2), dim=1)
            learned_Moire_cycle_64_2 = G_Artifact_64_2(deep_Demoire_Noise_64)
            Reconv_Moire_64_2 = Reconv_Moire_64_1.detach() + learned_Moire_cycle_64_2

            loss_cycle_64 = criterion_cycle(Reconv_Moire_64_2, MOIRE_64)

            loss_G_64 = (loss_GAN_64_1+loss_GAN_64_2) + 50 * loss_cycle_64

            # Backward + Optimize
            loss_G_64.backward()
            optimizer_G_64_1.step()
            optimizer_G_64_2.step()


            # ------------------
            #  Train Generators 128 size
            # ------------------
            # forward
            optimizer_G_128_2.zero_grad()
            G_Artifact_128_2.train()
            Net_Demoire_128.eval()

            # Use 64 TMB
            pseudo_Moire_128 = CLEAR_128 * learned_Moire_64.detach()

            # G2
            learned_Moire_64_128 = upx2(learned_Moire_64_2.detach())
            deep_real_Clean_Noise_128 = torch.cat((pseudo_Moire_128.detach(), learned_Moire_64_128), dim=1)
            learned_Moire_128_2 = G_Artifact_128_2(deep_real_Clean_Noise_128)
            deep_pseudo_Moire_128 = pseudo_Moire_128.detach() + learned_Moire_128_2

            deep_pseudo_Moire_filter_128 = deep_pseudo_Moire_128 - kornia.filters.median_blur(deep_pseudo_Moire_128, (3, 3))
            deep_pseudo_Moire_cat_128 = torch.cat((deep_pseudo_Moire_128, deep_pseudo_Moire_filter_128), dim=1)

            ## G2-GAN LOSS 2 ##
            loss_GAN_128_2 = criterion_GAN(D_Moire_128(deep_pseudo_Moire_cat_128), valid_128)

            # G2-CYCLE LOSS
            # Real Moire(256 size) -> DemoireNet -> Demoire image -> Downsampling for Multi-GAN
            fusion_64_128 = torch.cat((MOIRE_128, upx2(Demoire_64.detach())), dim=1)
            Demoire_128 = Net_Demoire_128(fusion_64_128).detach()

            final_Demoire_128 = Demoire_128 / Demoire_bright_alpha.detach()

            Reconv_Moire_128_1 = final_Demoire_128 * learned_Moire_2.detach()

            learned_Moire_cycle_64_2_128 = upx2(learned_Moire_cycle_64_2.detach())
            deep_Demoire_Noise_128 = torch.cat((Reconv_Moire_128_1, learned_Moire_cycle_64_2_128), dim=1)
            learned_Moire_cycle_128_2 = G_Artifact_128_2(deep_Demoire_Noise_128)
            Reconv_Moire_128_2 = Reconv_Moire_128_1.detach() + learned_Moire_cycle_128_2

            loss_cycle_128 = criterion_cycle(Reconv_Moire_128_2, MOIRE_128)

            loss_G_128 = loss_GAN_128_2 + 50 * loss_cycle_128

            # Backward + Optimize
            loss_G_128.backward()
            optimizer_G_128_2.step()

            # ------------------
            #  Train Generators 256 size
            # ------------------
            # forward
            optimizer_G_256_2.zero_grad()
            G_Artifact_256_2.train()
            Net_Demoire_256.eval()

            # Use 64 TMB
            pseudo_Moire_256 = CLEAR_256 * learned_Moire_64.detach()

            # Artifact 2
            learned_Moire_128_256 = upx2(learned_Moire_128_2.detach())
            deep_real_Clean_Noise_256 = torch.cat((pseudo_Moire_256.detach(), learned_Moire_128_256), dim=1)
            learned_Moire_256_2 = G_Artifact_256_2(deep_real_Clean_Noise_256)
            deep_pseudo_Moire_256 = pseudo_Moire_256.detach() + learned_Moire_256_2

            deep_pseudo_Moire_filter_256 = deep_pseudo_Moire_256 - kornia.filters.median_blur(deep_pseudo_Moire_256, (3, 3))
            deep_pseudo_Moire_cat_256 = torch.cat((deep_pseudo_Moire_256, deep_pseudo_Moire_filter_256), dim=1)

            ## G2-GAN LOSS 2 ##
            loss_GAN_256_2 = criterion_GAN(D_Moire_256(deep_pseudo_Moire_cat_256), valid_256)

            # G2-CYCLE LOSS
            # Real Moire(256 size) -> DemoireNet -> Demoire image -> Downsampling for Multi-GAN
            fusion_128_256 = torch.cat((MOIRE_256, upx2(Demoire_128.detach())), dim=1)
            Demoire_256 = Net_Demoire_256(fusion_128_256).detach()
            final_Demoire_256 = Demoire_256 / Demoire_bright_alpha.detach()

            Reconv_Moire_256_1 = final_Demoire_256 * learned_Moire_2.detach()

            learned_Moire_cycle_128_2_256 = upx2(learned_Moire_cycle_128_2.detach())
            deep_Demoire_Noise_256 = torch.cat((Reconv_Moire_256_1, learned_Moire_cycle_128_2_256), dim=1)
            learned_Moire_cycle_256_2 = G_Artifact_256_2(deep_Demoire_Noise_256)
            Reconv_Moire_256_2 = Reconv_Moire_256_1.detach() + learned_Moire_cycle_256_2

            loss_cycle_256 = criterion_cycle(Reconv_Moire_256_2, MOIRE_256)

            loss_G_256 = loss_GAN_256_2 + 50 * loss_cycle_256

            # Backward + Optimize
            loss_G_256.backward()
            optimizer_G_256_2.step()

            # ------------------
            #  Train Demoire network 64 size
            # ------------------
            optimizer_Net_Demoire_64.zero_grad()
            Net_Demoire_64.train()
            Net_Demoire_TMB.train()
            G_Artifact_64_2.eval()

            # Sample noise as generator input
            pseudo_Moire_64 = CLEAR_64 * learned_Moire_64.detach()

            # double generator
            z_64 = Variable(Tensor(np.random.uniform(-1, 1, size=CLEAR_64.shape).astype(np.float32)).to(cuda))
            deep_real_Clean_Noise_64 = torch.cat((pseudo_Moire_64.detach(), z_64), dim=1)
            learned_Moire_64_2 = G_Artifact_64_2(deep_real_Clean_Noise_64).detach()
            deep_pseudo_Moire_64 = pseudo_Moire_64.detach() + learned_Moire_64_2

            # Content loss
            reconv_clean_64 = Net_Demoire_64(deep_pseudo_Moire_64)

            # reconv_clean_64 HISTOGRAM
            A_1, B_1, C_1, D_1 = torch.chunk(reconv_clean_64, 4, dim=0)

            A_1 = torch.squeeze(A_1)
            B_1 = torch.squeeze(B_1)
            C_1 = torch.squeeze(C_1)
            D_1 = torch.squeeze(D_1)

            trans = torchvision.transforms.ToPILImage()

            A_1 = trans(A_1)
            B_1 = trans(B_1)
            C_1 = trans(C_1)
            D_1 = trans(D_1)

            A_GRAY_1 = transforms.Grayscale(num_output_channels=1)(A_1)
            B_GRAY_1 = transforms.Grayscale(num_output_channels=1)(B_1)
            C_GRAY_1 = transforms.Grayscale(num_output_channels=1)(C_1)
            D_GRAY_1 = transforms.Grayscale(num_output_channels=1)(D_1)

            A_gray_tensor_1 = torch.as_tensor(A_GRAY_1.histogram())
            B_gray_tensor_2 = torch.as_tensor(B_GRAY_1.histogram())
            C_gray_tensor_3 = torch.as_tensor(C_GRAY_1.histogram())
            D_gray_tensor_4 = torch.as_tensor(D_GRAY_1.histogram())

            reconv_clean_64_histogram = torch.stack(
                (A_gray_tensor_1, B_gray_tensor_2, C_gray_tensor_3, D_gray_tensor_4), dim=0)

            reconv_clean_64_histogram = reconv_clean_64_histogram / 64
            reconv_clean_64_histogram = reconv_clean_64_histogram.to(cuda)

            bright_alpha = Net_Demoire_TMB(reconv_clean_64_histogram)

            bright_alpha = bright_alpha + (1e-3)

            final_64 = reconv_clean_64 / bright_alpha

            loss_content_64 = criterion_content(final_64, CLEAR_64)

            # GAN LOSS
            Demoire_pattern_64 = Net_Demoire_64(MOIRE_64)

            # reconv_clean_64 HISTOGRAM
            a_1, b_1, c_1, d_1 = torch.chunk(Demoire_pattern_64, 4, dim=0)

            a_1 = torch.squeeze(a_1)
            b_1 = torch.squeeze(b_1)
            c_1 = torch.squeeze(c_1)
            d_1 = torch.squeeze(d_1)

            trans = torchvision.transforms.ToPILImage()

            a_1 = trans(a_1)
            b_1 = trans(b_1)
            c_1 = trans(c_1)
            d_1 = trans(d_1)

            a_GRAY_1 = transforms.Grayscale(num_output_channels=1)(a_1)
            b_GRAY_1 = transforms.Grayscale(num_output_channels=1)(b_1)
            c_GRAY_1 = transforms.Grayscale(num_output_channels=1)(c_1)
            d_GRAY_1 = transforms.Grayscale(num_output_channels=1)(d_1)

            a_gray_tensor_1 = torch.as_tensor(a_GRAY_1.histogram())
            b_gray_tensor_2 = torch.as_tensor(b_GRAY_1.histogram())
            c_gray_tensor_3 = torch.as_tensor(c_GRAY_1.histogram())
            d_gray_tensor_4 = torch.as_tensor(d_GRAY_1.histogram())

            Demoire_pattern_64_histogram = torch.stack(
                (a_gray_tensor_1, b_gray_tensor_2, c_gray_tensor_3, d_gray_tensor_4), dim=0)

            Demoire_pattern_64_histogram = Demoire_pattern_64_histogram / 64
            Demoire_pattern_64_histogram = Demoire_pattern_64_histogram.to(cuda)

            cycle_bright_alpha = Net_Demoire_TMB(Demoire_pattern_64_histogram)

            cycle_bright_alpha = cycle_bright_alpha + (1e-3)

            Demoire_64 = Demoire_pattern_64 / cycle_bright_alpha

            Demoire_filter_64 = Demoire_64 - kornia.filters.median_blur(Demoire_64, (3, 3))
            Demoire_filter_64_cat = torch.cat((Demoire_64, Demoire_filter_64), dim=1)

            loss_GAN_64 = criterion_GAN(D_Clear_64(Demoire_filter_64_cat), valid_64)

            # Total loss
            loss_Net_64 = loss_GAN_64 + 50 * loss_content_64

            # Backward + Optimize
            loss_Net_64.backward()
            optimizer_Net_Demoire_TMB.step()
            optimizer_Net_Demoire_64.step()

            # ------------------
            #  Train Demoire network 128 size
            # ------------------
            optimizer_Net_Demoire_128.zero_grad()
            Net_Demoire_128.train()
            G_Artifact_128_2.eval()

            # Sample noise as generator input
            pseudo_Moire_128 = CLEAR_128 * learned_Moire_64.detach()

            # double generator
            z_128 = Variable(Tensor(np.random.uniform(-1, 1, size=CLEAR_128.shape).astype(np.float32)).to(cuda))
            deep_real_Clean_Noise_128 = torch.cat((pseudo_Moire_128.detach(), z_128), dim=1)
            learned_Moire_128_2 = G_Artifact_128_2(deep_real_Clean_Noise_128).detach()
            deep_pseudo_Moire_128 = pseudo_Moire_128.detach() + learned_Moire_128_2

            # Content loss
            gradual_fusion_64_128 = torch.cat((deep_pseudo_Moire_128.detach(), upx2(reconv_clean_64.detach())), dim=1)
            reconv_clean_128 = Net_Demoire_128(gradual_fusion_64_128)

            final_128 = reconv_clean_128 / bright_alpha.detach()

            loss_content_128 = criterion_content(final_128, CLEAR_128)

            # GAN LOSS
            cycle_gradual_fusion_64_128 = torch.cat((MOIRE_128, upx2(Demoire_64.detach())), dim=1)
            Demoire_pattern_128 = Net_Demoire_128(cycle_gradual_fusion_64_128)

            Demoire_128 = Demoire_pattern_128 / cycle_bright_alpha.detach()

            Demoire_filter_128 = Demoire_128 - kornia.filters.median_blur(Demoire_128, (3, 3))
            Demoire_filter_128_cat = torch.cat((Demoire_128, Demoire_filter_128), dim=1)

            loss_GAN_128 = criterion_GAN(D_Clear_128(Demoire_filter_128_cat), valid_128)

            # Total loss
            loss_Net_128 = loss_GAN_128 + 50 * loss_content_128

            # Backward + Optimize
            loss_Net_128.backward()
            optimizer_Net_Demoire_128.step()

            # ------------------
            #  Train Demoire network 256 size
            # ------------------
            optimizer_Net_Demoire_256.zero_grad()
            Net_Demoire_256.train()
            G_Artifact_256_2.eval()

            # Sample noise as generator input
            pseudo_Moire_256 = CLEAR_256 * learned_Moire_64.detach()

            # double generator
            z_256 = Variable(Tensor(np.random.uniform(-1, 1, size=CLEAR_256.shape).astype(np.float32)).to(cuda))
            deep_real_Clean_Noise_256 = torch.cat((pseudo_Moire_256.detach(), z_256), dim=1)
            learned_Moire_256_2 = G_Artifact_256_2(deep_real_Clean_Noise_256).detach()
            deep_pseudo_Moire_256 = pseudo_Moire_256.detach() + learned_Moire_256_2

            # Content loss
            gradual_fusion_128_256 = torch.cat((deep_pseudo_Moire_256.detach(), upx2(reconv_clean_128.detach())), dim=1)
            reconv_clean_256 = Net_Demoire_256(gradual_fusion_128_256)

            final_256 = reconv_clean_256 / bright_alpha.detach()

            loss_content_256 = criterion_content(final_256, CLEAR_256)

            # GAN LOSS
            cycle_gradual_fusion_128_256 = torch.cat((MOIRE_256, upx2(Demoire_128.detach())), dim=1)
            Demoire_pattern_256 = Net_Demoire_256(cycle_gradual_fusion_128_256)

            Demoire_256 = Demoire_pattern_256 / cycle_bright_alpha.detach()

            Demoire_filter_256 = Demoire_256 - kornia.filters.median_blur(Demoire_256, (3, 3))
            Demoire_filter_256_cat = torch.cat((Demoire_256, Demoire_filter_256), dim=1)

            loss_GAN_256 = criterion_GAN(D_Clear_256(Demoire_filter_256_cat), valid_256)

            # Total loss
            loss_Net_256 = loss_GAN_256 + 50 * loss_content_256

            # Backward + Optimize
            loss_Net_256.backward()
            optimizer_Net_Demoire_256.step()

            # ------------------
            #  Train PIXEL Moire Discriminator
            # ------------------
            # 64
            optimizer_D_Moire_64.zero_grad()
            D_Moire_64.train()

            # Real loss 64 size
            real_Moire_filter_64 = MOIRE_64 - kornia.filters.median_blur(MOIRE_64, (3, 3))
            real_Moire_cat_64 = torch.cat((MOIRE_64, real_Moire_filter_64), dim=1)

            real_loss_64 = criterion_GAN(D_Moire_64(real_Moire_cat_64), valid_64)

            # Fake loss (on batch of previously generated samples)
            fake_loss_64 = criterion_GAN(D_Moire_64(pseudo_Moire_cat_64.detach()), fake_64)

            # Fake loss (on batch of previously generated samples)
            fake2_loss_64 = criterion_GAN(D_Moire_64(deep_pseudo_Moire_cat_64.detach()), fake_64)

            loss_pixel_D_64 = (real_loss_64 + fake_loss_64) / 2 + (real_loss_64 + fake2_loss_64) / 2

            loss_pixel_D_64.backward()
            optimizer_D_Moire_64.step()

            # 128
            optimizer_D_Moire_128.zero_grad()
            D_Moire_128.train()

            # Real loss 128 size
            real_Moire_filter_128 = MOIRE_128 - kornia.filters.median_blur(MOIRE_128, (3, 3))
            real_Moire_cat_128 = torch.cat((MOIRE_128, real_Moire_filter_128), dim=1)

            real_loss_128 = criterion_GAN(D_Moire_128(real_Moire_cat_128), valid_128)

            # Fake loss (on batch of previously generated samples)
            fake2_loss_128 = criterion_GAN(D_Moire_128(deep_pseudo_Moire_cat_128.detach()), fake_128)

            loss_pixel_D_128 = (real_loss_128 + fake2_loss_128) / 2

            loss_pixel_D_128.backward()
            optimizer_D_Moire_128.step()

            # 256
            optimizer_D_Moire_256.zero_grad()
            D_Moire_256.train()

            # Real loss 256 size
            real_Moire_filter_256 = MOIRE_256 - kornia.filters.median_blur(MOIRE_256, (3, 3))
            real_Moire_cat_256 = torch.cat((MOIRE_256, real_Moire_filter_256), dim=1)

            real_loss_256 = criterion_GAN(D_Moire_256(real_Moire_cat_256), valid_256)

            # Fake loss (on batch of previously generated samples)
            fake2_loss_256 = criterion_GAN(D_Moire_256(deep_pseudo_Moire_cat_256.detach()), fake_256)

            loss_pixel_D_256 = (real_loss_256 + fake2_loss_256) / 2

            loss_pixel_D_256.backward()
            optimizer_D_Moire_256.step()

            loss_D_Moire = (loss_pixel_D_64 + loss_pixel_D_128 + loss_pixel_D_256) / 3

            # ------------------
            #  Train PIXEL Clear Discriminator
            # ------------------
            # 64
            optimizer_D_Clear_64.zero_grad()
            D_Clear_64.train()

            # Real loss 64 size
            real_Clean_filter_64 = CLEAR_64 - kornia.filters.median_blur(CLEAR_64, (3, 3))
            real_Clean_cat_64 = torch.cat((CLEAR_64, real_Clean_filter_64), dim=1)

            real_loss_64 = criterion_GAN(D_Clear_64(real_Clean_cat_64), valid_64)

            # Fake loss (on batch of previously generated samples)
            fake_loss_64 = criterion_GAN(D_Clear_64(Demoire_filter_64_cat.detach()), fake_64)

            loss_D_clear_64 = (real_loss_64 + fake_loss_64) / 2

            # Backward + Optimize
            loss_D_clear_64.backward()
            optimizer_D_Clear_64.step()

            # 128
            optimizer_D_Clear_128.zero_grad()
            D_Clear_128.train()

            # Real loss 128 size
            real_Clean_filter_128 = CLEAR_128 - kornia.filters.median_blur(CLEAR_128, (3, 3))
            real_Clean_cat_128 = torch.cat((CLEAR_128, real_Clean_filter_128), dim=1)

            real_loss_128 = criterion_GAN(D_Clear_128(real_Clean_cat_128), valid_128)

            # Fake loss (on batch of previously generated samples)
            fake_loss_128 = criterion_GAN(D_Clear_128(Demoire_filter_128_cat.detach()), fake_128)

            loss_D_clear_128 = (real_loss_128 + fake_loss_128) / 2

            # Backward + Optimize
            loss_D_clear_128.backward()
            optimizer_D_Clear_128.step()

            # 256
            optimizer_D_Clear_256.zero_grad()
            D_Clear_256.train()

            # Real loss 256 size
            real_Clean_filter_256 = CLEAR_256 - kornia.filters.median_blur(CLEAR_256, (3, 3))
            real_Clean_cat_256 = torch.cat((CLEAR_256, real_Clean_filter_256), dim=1)

            real_loss_256 = criterion_GAN(D_Clear_256(real_Clean_cat_256), valid_256)

            # Fake loss (on batch of previously generated samples)
            fake_loss_256 = criterion_GAN(D_Clear_256(Demoire_filter_256_cat.detach()), fake_256)

            loss_D_clear_256 = (real_loss_256 + fake_loss_256) / 2

            # Backward + Optimize
            loss_D_clear_256.backward()
            optimizer_D_Clear_256.step()

            loss_D_clear = (loss_D_clear_64 + loss_D_clear_128 + loss_D_clear_256) / 3

            loss_G = loss_G_64 + loss_G_128 + loss_G_256
            loss_D = loss_D_clear + loss_D_Moire
            loss_content = loss_Net_256 + loss_Net_128 + loss_Net_64
            # Display loss value
            if iteration % 100 == 0:
                print(
                    '[%d/%d][%d/%d]\t NET-LOSS - L1-ASL: %.4f  G-LOSS - G: %.4f  D-LOSS - D: %.4f' % (
                        epoch, NUM_EPOCHS, iteration, len(data_train_loader), loss_content, loss_G, loss_D))

            iteration += 1

        # Save after each epoch
        # if (epoch + 1) % 10 == 0:
        torch.save(Net_Demoire_256.state_dict(), 'MODEL/%s/Net_Demoire_256/Net_Demoire_256_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(Net_Demoire_128.state_dict(), 'MODEL/%s/Net_Demoire_128/Net_Demoire_128_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(Net_Demoire_64.state_dict(), 'MODEL/%s/Net_Demoire_64/Net_Demoire_64_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(Net_Demoire_TMB.state_dict(), 'MODEL/%s/Net_Demoire_TMB/Net_Demoire_TMB_%d.pth' % (opt.save_data_name, epoch + 1))

        torch.save(G_Artifact_256_2.state_dict(), 'MODEL/%s/G_Artifact_256_2/G_Artifact_256_2_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(D_Moire_256.state_dict(), 'MODEL/%s/D_Moire_256/D_Moire_256_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(D_Clear_256.state_dict(), 'MODEL/%s/D_Clear_256/D_Clear_256_%d.pth' % (opt.save_data_name, epoch + 1))

        torch.save(G_Artifact_128_2.state_dict(), 'MODEL/%s/G_Artifact_128_2/G_Artifact_128_2_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(D_Moire_128.state_dict(), 'MODEL/%s/D_Moire_128/D_Moire_128_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(D_Clear_128.state_dict(), 'MODEL/%s/D_Clear_128/D_Clear_128_%d.pth' % (opt.save_data_name, epoch + 1))

        torch.save(G_Artifact_64_1.state_dict(), 'MODEL/%s/G_Artifact_64_1/G_Artifact_64_1_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(G_Artifact_64_2.state_dict(), 'MODEL/%s/G_Artifact_64_2/G_Artifact_64_2_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(D_Moire_64.state_dict(), 'MODEL/%s/D_Moire_64/D_Moire_64_%d.pth' % (opt.save_data_name, epoch + 1))
        torch.save(D_Clear_64.state_dict(), 'MODEL/%s/D_Clear_64/D_Clear_64_%d.pth' % (opt.save_data_name, epoch + 1))


	    # Step LR UPDATE
        lr_scheduler_Net_256.step()
        lr_scheduler_Net_128.step()
        lr_scheduler_Net_64.step()
        lr_scheduler_Net_Demoire_TMB.step()

        lr_scheduler_G_256_2.step()
        lr_scheduler_D_Moire_256.step()
        lr_scheduler_D_Clear_256.step()

        lr_scheduler_G_128_2.step()
        lr_scheduler_D_Moire_128.step()
        lr_scheduler_D_Clear_128.step()

        lr_scheduler_G_64_1.step()
        lr_scheduler_G_64_2.step()
        lr_scheduler_D_Moire_64.step()
        lr_scheduler_D_Clear_64.step()


        # Validation dataset root
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
                save_image(image_grid, "MODEL/%s/IMAGE/%s/%s.png" % (opt.save_data_name,epoch , i), normalize=False)


            print("VALIDATION PSNR: ===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(in_path)))
            print("VALIDATION SSIM: ===> Avg. SSIM: {:.4f} dB".format(score))
            tfb_train_writer.add_scalar('TEST-PSNR_1024', np.average(avg_psnr / len(in_path)), global_step=epoch,
                                        walltime=time.time())
            tfb_train_writer.add_scalar('TEST-SSIM_1024', np.average(score), global_step=epoch,
                                        walltime=time.time())
            tfb_train_writer.add_scalar('TRAIN-GENERATOR-LOSS', np.average(loss_G.cpu().detach().numpy()),
                                        global_step=epoch, \
                                        walltime=time.time())
            tfb_train_writer.add_scalar('TRAIN-DISCRIMINATOR-LOSS', np.average(loss_D.cpu().detach().numpy()),
                                        global_step=epoch, \
                                        walltime=time.time())
