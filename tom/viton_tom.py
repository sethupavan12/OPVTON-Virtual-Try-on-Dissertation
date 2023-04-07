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
from bpgm.model.models import BPGM
from bpgm.model.utils import load_checkpoint, save_checkpoint
from bpgm.dataset import DataLoader, VitonDataset
from bpgm.utils.loss import VGGLoss, SSIMLoss
from bpgm.utils.visualization import board_add_images


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
                                                  max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()

        agnostic = inputs['body_image'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        # display array shape
        print("agnostic shape: ", agnostic.shape)
        print("c shape: ", c.shape)
        print("cm shape: ", cm.shape)

        outputs = model(torch.cat([agnostic, c, cm], 1))  # CHANGED

        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [[c, cm*2-1, m_composite*2-1],  # CHANGED
                   [p_rendered, p_tryon, im]]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)  # CHANGED
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

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

    # CHANGED: Updated input channels from 26 to 28
    model = UnetGenerator(28, 4, 6, ngf=64, norm_layer=CustomInstanceNorm2d)  # CP-VTON+

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    train_tom(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(
        opt.checkpoint_dir, opt.name, 'tom_final.pth'))

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


class UnetGenerator(nn.Module):
    def __init__(self, num_downs, input_nc=7, output_nc=7, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=input_nc, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)







# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == CustomInstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,  # Change kernel_size to 3
                            stride=2, padding=1, bias=use_bias) if not outermost else nn.Conv2d(7, inner_nc, kernel_size=3,  # Change kernel_size to 3
                            stride=2, padding=1, bias=use_bias)  # CHANGED: input channel size for outermost

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc,kernel_size=3, stride=1, padding=1, bias=use_bias)  # Change kernel_size to 3
            down = [downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3,stride=1, padding=1, bias=use_bias)  # Change kernel_size to 3
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3,stride=1, padding=1, bias=use_bias)  # Change kernel_size to 3
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)





def get_opt():
    parser = argparse.ArgumentParser()
    # Name of the GMM or TOM model
    parser.add_argument("--name", default="GMM")
    # parser.add_argument("--name", default="TOM")

    # Add multiple workers support
    parser.add_argument("--workers", type=int, default=mp.cpu_count() // 2)


    # GPU IDs to use
    # parser.add_argument("--gpu_ids", default="")


    # Number of workers for dataloader (default: 1)
    #parser.add_argument('-j', '--workers', type=int, default=1)
    # Batch size for training (default: 32)
    # Batch size defines the number of images that are processed at the same time
    parser.add_argument('-b', '--batch-size', type=int, default=32)

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


class CustomInstanceNorm2d(nn.InstanceNorm2d):
    def forward(self, input):
        if input.size(2) == 1 and input.size(3) == 1:
            return input
        return super().forward(input)

# Replace all instances of CustomInstanceNorm2d with CustomInstanceNorm2d in your model



if __name__ == "__main__":
    main()
