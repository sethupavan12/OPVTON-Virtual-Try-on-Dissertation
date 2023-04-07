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
from PIL import Image

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class VirtualTryOnUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.down1 = UNetDown(7, 64) # 3 (agnostic) + 3 (cloth) + 1 (cloth_mask)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)

        # Decoder
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(256 * 2, 128)
        self.up3 = UNetUp(128 * 2, 64)

        # Output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 * 2, 4, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)

        output = self.final(u3)

        p_rendered, m_composite = torch.split(output, 3, 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)

        return p_rendered, m_composite


def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    #criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()

        # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
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

        # Forward pass
        p_rendered, m_composite = model(torch.cat([agnostic, c, cm], 1))

        # Compute p_tryon
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        # Calculate losses
        loss_l1 = criterionL1(p_tryon, im)
        perceptual_loss = loss_fn_alex(p_tryon, im)
        #loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + perceptual_loss + loss_mask

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
            board.add_scalar('Mask', loss_mask.item(), step)
            board.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                  % (step+1, t, loss.item(), loss_l1.item(),
                     loss_vgg.item(), loss_mask.item()), flush=True)

        # Save checkpoints
        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%07d.pth' % (step + 1)))
    

        # save checkpoint at the 50th step
        if step == 21:
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



if __name__ == "__main__":
    main()



# tensor = torch.rand(1, 3, 256, 192)
# tensor = tensor.permute(0, 2, 3, 1).cpu().detach().numpy()
# image = Image.fromarray((tensor[0] * 255).astype('uint8'))

# # Save the image
# image.save('tensor_image.png')
