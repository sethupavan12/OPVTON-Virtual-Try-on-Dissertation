import sys

sys.path.append('../')

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import argparse
import multiprocessing as mp
import lpips
# Import all the things we need for the model
from bpgm.model.utils import load_checkpoint, save_checkpoint
from bpgm.dataset import DataLoader, VitonDataset
from bpgm.utils.loss import VGGLoss, SSIMLoss
from bpgm.utils.visualization import board_add_images
from PIL import Image
from pytorch_msssim import ssim
import torchvision


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.conv = DoubleConv(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VirtualTryOnUNet(nn.Module):
    def __init__(self, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(18, 64) # 3 (agnostic) + 3 (cloth) + 1 (cloth_mask) # total 18 if segmentation
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output
        self.outc = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Tanh()
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.dropout(x)
        output = self.outc(x)

        p_rendered, m_composite = torch.split(output, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)

        return p_rendered, m_composite


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    discriminator = PatchGANDiscriminator(21).cuda()


    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    criterionGAN = nn.BCEWithLogitsLoss()


    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))


    
    # In the training loop
    for step in range(opt.keep_step + opt.decay_step):
        
        iter_start_time = time.time()
        inputs = train_loader.next_batch()
        im = inputs['image'].cuda()
        agnostic = inputs['body_image'].cuda()
        # save body image for debugging as a pictur
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        seg = inputs['body_label'].cuda()

        # Forward pass
        p_rendered, m_composite = model(torch.cat([agnostic, c, cm,seg], 1))

        # Compute p_tryon
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)


        # Discriminator update
        real_input = torch.cat([agnostic, c, cm, im, seg], 1)
        fake_input = torch.cat([agnostic, c, cm, p_tryon.detach(), seg], 1)


        real_output = discriminator(real_input)
        fake_output = discriminator(fake_input)

        real_labels = torch.ones_like(real_output).cuda()
        fake_labels = torch.zeros_like(fake_output).cuda()

        lossD_real = criterionGAN(real_output, real_labels)
        lossD_fake = criterionGAN(fake_output, fake_labels)

        lossD = (lossD_real + lossD_fake) * 0.5

        optimizerD.zero_grad()
        lossD.backward(retain_graph=True)
        optimizerD.step()

        # Generator update
        fake_output = discriminator(fake_input)

        # Calculate losses
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        ssim_loss = 1 - ssim( p_tryon, im, data_range=255, size_average=True)
        lossG = criterionGAN(fake_output, real_labels)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask + 0.1 * ssim_loss + lossG

        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update learning rate scheduler
        scheduler.step()

        visuals = [[agnostic, c, cm],
            [c, cm*2-1, m_composite*2-1],
            [p_rendered, p_tryon, im]]

        # Logging and visualization
        if step % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step)
            board.add_scalar('metric', loss.item(), step)
            board.add_scalar('L1', loss_l1.item(), step)
            board.add_scalar('VGG', loss_vgg.item(), step)
            board.add_scalar('lossD', lossD.item(), step)
            board.add_scalar('lossG', lossG.item(), step)
            board.add_scalar('ssim_loss', ssim_loss, step)
            board.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, ssim_loss: %.4f , lossD: %.4f, lossG: %.4f'
                % (step+1, t, loss.item(), loss_l1.item(),
                    loss_vgg.item(), ssim_loss.item(), lossD.item(), lossG.item()), flush=True)


        # Save checkpoints
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%07d.pth' % (step + 1)))
    



def main():
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)

    # create dataset
    if opt.dataset == "viton":
        train_dataset = VitonDataset(opt)
    else:
        raise NotImplementedError

    # create dataloader
    train_loader = DataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    model = VirtualTryOnUNet().cuda()

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    train_tom(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(
        opt.checkpoint_dir, opt.name, 'tom_final.pth'))

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))



def get_opt():
    parser = argparse.ArgumentParser()
    # Name of the GMM or TOM model
    parser.add_argument("--name", default="TOM_with_CPVTON_LIKE_BUT_GAN")
    # parser.add_argument("--name", default="TOM")

    # Add multiple workers support
    parser.add_argument("--workers", type=int, default=mp.cpu_count() // 2)


    # GPU IDs to use
    # parser.add_argument("--gpu_ids", default="")


    # Number of workers for dataloader (default: 1)
    #parser.add_argument('-j', '--workers', type=int, default=1)
    # Batch size for training (default: 32)
    # Batch size defines the number of images that are processed at the same time
    parser.add_argument('-b', '--batch-size', type=int, default=16)

    # Path to the data folder
    parser.add_argument("--dataroot", default="/scratch/c.c1984628/my_diss/bpgm/data")

    # Training mode or testing mode
    parser.add_argument("--datamode", default="train")

    # What are we training/testing here
    parser.add_argument("--stage", default="TOM")
    # parser.add_argument("--stage", default="TOM")

    # Path to the list of training/testing images
    parser.add_argument("--data_list", default="/scratch/c.c1984628/my_diss/bpgm/data/train_pairs.txt")

    # choose dataset
    parser.add_argument("--dataset", default="viton")

    # fine_width, fine_height: size of the input image to the network
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)

    # lr = learning rate
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    # tensorboard_dir: path to the folder where tensorboard files are saved
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization')
    # display_count: how often to display the training results defaulted to every 20 steps
    parser.add_argument("--display_count", type=int, default=20)
    # save_count: how often to save the model defaulted to every 5000 steps
    parser.add_argument("--save_count", type=int, default=5000)
    # keep_step: how many steps to train the model for
    parser.add_argument("--keep_step", type=int, default=100000)
    # decay_step: how many steps to decay the learning rate for
    parser.add_argument("--decay_step", type=int, default=100000)
    # shuffle: shuffle the input data
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt



if __name__ == "__main__":
    main()


