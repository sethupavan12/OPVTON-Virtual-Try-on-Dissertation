
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
import torchvision


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalization=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalization:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, use_batch_norm=True):
            block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.ReLU(inplace=True))
            return block

        def deconv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, use_batch_norm=True):
            block = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False)]
            if use_batch_norm:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.ReLU(inplace=True))
            return block

        self.encoder = nn.Sequential(
            *conv_block(18, 64, use_batch_norm=False),
            *conv_block(64, 128),
            *conv_block(128, 256),
            *conv_block(256, 512),
            *conv_block(512, 512)
        )

        self.decoder = nn.Sequential(
            *deconv_block(512, 512),
            *deconv_block(512, 256),
            *deconv_block(256, 128),
            *deconv_block(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns layers of each discriminator block."""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(9, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, img_input):
        # Concatenate image and condition image by channels to produce input
        return self.model(img_input)

def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    discriminator = Discriminator().cuda()
    

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    criterionGAN = nn.BCELoss()

        # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))


    # Training loop
    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()
        agnostic = inputs['body_image'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        seg = inputs['body_label'].cuda()

        # Forward pass
        # p_tryon = model(torch.cat([agnostic, c, cm], 1))


        p_tryon = model(torch.cat([agnostic, c, cm,seg], 1))


        # Calculate losses
        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        # convert the perceptual loss to a scalar
        loss_perceptual = loss_fn_alex.forward(p_tryon, im)
        loss_perceptual = torch.mean(loss_perceptual)

        gen_fake_decision = discriminator(torch.cat([p_tryon, agnostic, c], 1)) #FORGOT TO ADD SEG!!
        # Adversarial loss for the generator
        real_label = torch.ones(gen_fake_decision.size()).cuda()
        fake_label = torch.zeros(gen_fake_decision.size()).cuda()
        
        loss_gen_adv = criterionGAN(gen_fake_decision, real_label)

        # Combine the losses and weight the perceptual and adversarial losses (use your desired weights)
        perceptual_weight = 0.2
        adv_weight = 0.5
        loss = loss_l1 + loss_vgg + perceptual_weight * loss_perceptual + adv_weight * loss_gen_adv

        # Update generator weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train the discriminator
        d_optimizer.zero_grad()

        # Real image
        real_decision = discriminator(torch.cat([im, agnostic, c], 1))
        loss_real = criterionGAN(real_decision, real_label)

        # Fake image
        with torch.no_grad():
            p_tryon_detached = p_tryon.detach()
        fake_decision = discriminator(torch.cat([p_tryon_detached, agnostic, c], 1))
        loss_fake = criterionGAN(fake_decision, fake_label)

        # Calculate the total discriminator loss
        loss_d = (loss_real + loss_fake) * 0.5

        # Update discriminator weights
        loss_d.backward()
        d_optimizer.step()

        # Optional: Log losses, update learning rate scheduler, etc.
        if step % opt.display_count == 0:
            print("Step: [%d/%d] Loss: %.4f (L1: %.4f, VGG: %.4f, Perceptual: %.4f, Adv: %.4f)" %
                (step, opt.keep_step + opt.decay_step, loss.item(), loss_l1.item(), loss_vgg.item(),
                perceptual_weight * loss_perceptual, adv_weight * loss_gen_adv.item()))
            print("Discriminator Loss: %.4f (Real: %.4f, Fake: %.4f)" % (loss_d.item(), loss_real.item(), loss_fake.item()))

        # every 50k steps, save the model
        if step % 50000 == 0:
            torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, opt.name,'step_%06d.pth' % step))
            torch.save(discriminator.state_dict(), os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d_d.pth' % step))


        # Tensorboard logging
        
        if (step) % opt.display_count == 0:
            label = inputs['label']
            visuals = [[agnostic, c, im], [label, cm, c], [p_tryon, im, agnostic]]

            # Add the images to the tensorboard
            board_add_images(board, 'combine', visuals, step)
            board.add_scalar('metric', loss.item(), step)
            board.add_scalar('VGG', loss_vgg.item(), step)
            board.add_scalar('L1', loss_l1.item(), step)
            board.add_scalar('perp', loss_perceptual, step)
            board.add_scalar('Discriminator Loss', loss_d.item(), step)
            board.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, perp: %.4f, vgg: %.4f'
                % (step+1, t, loss.item(), loss_l1.item(),
                    loss_perceptual, loss_vgg.item()), flush=True)


        # Update learning rate scheduler
        scheduler.step()



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

    model = Generator().cuda()
    #model = VirtualTryOnUNet().cuda()

    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    train_tom(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(
        opt.checkpoint_dir, opt.name, 'tom_adv_loss_final.pth'))

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))



def get_opt():
    parser = argparse.ArgumentParser()
    # Name of the GMM or TOM model
    parser.add_argument("--name", default="TOM_with_adversarial_loss_complex_with_seg_higher_batch_size_16")
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

   

