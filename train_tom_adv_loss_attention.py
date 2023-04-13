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
from torchvision.models import resnet50

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size, c, h, w = x.size()

        q = self.conv1(x)
        k = self.conv2(x)

        q = q.view(batch_size, self.out_channels, -1)
        k = k.view(batch_size, self.out_channels, -1).permute(0, 2, 1)
        v = x.view(batch_size, self.out_channels, -1)  # Change this line to use self.out_channels instead of c

        attention = torch.matmul(q, k)
        attention = F.softmax(attention, dim=-1)

        attention_v = torch.matmul(attention, v).view(batch_size, c, h, w)
        out = x + attention_v
        return out





class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = self._make_layer(64, 128, 2)
        self.conv3 = self._make_layer(128, 256, 3)
        self.conv4 = self._make_layer(256, 512, 4)
        self.conv5 = self._make_layer(512, 1024, 2)
        self.attention = Attention(1024, 192)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.attention(x)
        return x

    def _make_layer(self, in_channels, out_channels, n_blocks):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_blocks - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention = Attention(1024, 192)
        self.upconv5 = self._make_up_layer(1024, 512, 2)
        self.upconv4 = self._make_up_layer(512, 256, 4)
        self.upconv3 = self._make_up_layer(256, 128, 3)
        self.upconv2 = self._make_up_layer(128, 64, 2)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.attention(x)
        x = self.upconv5(x)
        x = self.upconv4(x)
        x = self.upconv3(x)
        x = self.upconv2(x)
        x = self.final_upsample(x)
        return x

    def _make_up_layer(self, in_channels, out_channels, n_blocks):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        for _ in range(n_blocks - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)



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
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(UNetUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        # Update the channel dimensions of UNetDoubleConv
        self.conv = UNetDoubleConv(in_channels, out_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Padding if the dimensions are not equal
        diff_height = x2.size()[2] - x1.size()[2]
        diff_width = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_width // 2, diff_width - diff_width // 2, diff_height // 2, diff_height - diff_height // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Change the number of input channels to the attention layer
        self.encoder = Encoder(18)
        self.middle = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x




# class Discriminator(nn.Module):
#     def __init__(self, in_channels=20):
#         super(Discriminator, self).__init__()
#         resnet = resnet50(pretrained=True)
        
#         # Replace the first convolutional layer in resnet
#         resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
#         self.pretrained = nn.Sequential(*list(resnet.children())[:-1])

#         self.final_conv = nn.Sequential(
#             nn.Conv2d(2048, 1, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, img_input):
#         x = self.pretrained(img_input)
#         x = self.final_conv(x)
#         return x.view(x.size(0), -1)
import torch
import torch.nn as nn
from torchvision.models import resnet50


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=20, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        resnet = resnet50(pretrained=True)

        # Replace the first convolutional layer in resnet
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.pretrained = nn.Sequential(*list(resnet.children())[:-1])

        self.patchgan = nn.Sequential(
            nn.Conv2d(2048, ndf * 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img_input):
        x = self.pretrained(img_input)
        x = self.patchgan(x)
        return x

def total_variation_loss(x):
    tv_h = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w



def train_tom(opt, train_loader, model, board):
    model.cuda()
    model.train()
    discriminator = PatchGANDiscriminator().cuda()
    

    # criterion
    criterionL1 = nn.L1Loss()
    #criterionL2 = nn.MSELoss() 
    #criterionVGG = VGGLoss()
    #criterionMask = nn.L1Loss()
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
        # print("agnostic shape:", agnostic.shape)
        # print("c shape:", c.shape)
        # print("cm shape:", cm.shape)
        # print("seg shape:" , seg.shape)
        p_tryon = model(torch.cat([agnostic, c, cm,seg], 1))

        # Calculate losses
        loss_l1 = criterionL1(p_tryon, im)
        loss_l2 = criterionL2(p_tryon, im)
        #loss_vgg = criterionVGG(p_tryon, im)
        # convert the perceptual loss to a scalar
        loss_perceptual = loss_fn_alex.forward(p_tryon, im)
        loss_perceptual = torch.mean(loss_perceptual)

        gen_fake_decision = discriminator(torch.cat([p_tryon, agnostic, c,seg], 1))
        # Adversarial loss for the generator
        real_label = torch.ones(gen_fake_decision.size(0),1).cuda()
        fake_label = torch.zeros(gen_fake_decision.size(0),1).cuda()
        
        loss_gen_adv = criterionGAN(gen_fake_decision.view(-1), real_label.view(-1))

        # Combine the losses and weight the perceptual and adversarial losses (use your desired weights)
        #perceptual_weight = 0.
        loss_perceptual = 5*loss_perceptual
        adv_weight = 0.1 # used to be 0.1
        loss = loss_l1 + loss_l2 + loss_perceptual + adv_weight * loss_gen_adv 

        # Update generator weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Train the discriminator
        d_optimizer.zero_grad()

        # Real image
        real_decision = discriminator(torch.cat([im, agnostic, c,seg], 1))
        loss_real = criterionGAN(real_decision.view(-1), real_label.view(-1))

        # Fake image
        with torch.no_grad():
            p_tryon_detached = p_tryon.detach()
        fake_decision = discriminator(torch.cat([p_tryon_detached, agnostic, c, seg], 1))
        loss_fake = criterionGAN(fake_decision.view(-1), fake_label.view(-1))

        # Calculate the total discriminator loss
        loss_d = (loss_real + loss_fake) * 0.5

        # Update discriminator weights
        loss_d.backward()
        d_optimizer.step()

        # Optional: Log losses, update learning rate scheduler, etc.
        if step % opt.display_count == 0:
            print("Step: [%d/%d] Loss: %.6f (L1: %.6f, L2: %.6f, Perceptual: %.6f, Adv: %.6f)" %
                (step, opt.keep_step + opt.decay_step, loss.item(), loss_l1.item(), loss_l2.item(),
                loss_perceptual, adv_weight * loss_gen_adv.item()))
            print("Discriminator Loss: %.6f (Real: %.6f, Fake: %.6f)" % (loss_d.item(), loss_real.item(), loss_fake.item()))

        # every 50k steps, save the model
        if step % 50000 == 0:
            # used to be /scratch/c.c1984628/my_diss/checkpoints/TOM_with_adv_loss_attention_patchGAN
            torch.save(model.state_dict(), os.path.join("/scratch/c.c1984628/my_diss/checkpoints/TOM_with_adv_loss_attention_patchGAN_HIGHER_ADV_LOSS_HIGHER_TRAIN",'step_%06d.pth' % step))
            torch.save(discriminator.state_dict(), os.path.join("/scratch/c.c1984628/my_diss/checkpoints/TOM_with_adv_loss_attention_patchGAN_HIGHER_ADV_LOSS_HIGHER_TRAIN", 'step_%06d_d.pth' % step))


        # Tensorboard logging
        
        
        if (step) % opt.display_count == 0:
            label = inputs['label']
            visuals = [[agnostic, c, im], [label, cm, c], [p_tryon, im, agnostic]]

            # Add the images to the tensorboard
            board_add_images(board, 'combine', visuals, step)
            board.add_scalar('metric', loss.item(), step)
            #board.add_scalar('VGG', loss_vgg.item(), step)
            board.add_scalar('L1', loss_l1.item(), step)
            board.add_scalar('perp', loss_perceptual, step)
            board.add_scalar('Discriminator Loss', loss_d.item(), step)
            board.add_scalar('Adv', adv_weight * loss_gen_adv.item(), step)
            board.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.6f, l1: %.6f, l2: %.6f, perp: %.6f'
                % (step+1, t, loss.item(), loss_l1.item(), loss_l2.item(),
                    loss_perceptual), flush=True)


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
    parser.add_argument("--name", default="TOM_with_adversarial_loss_attention_patchGAN")
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
    parser.add_argument("--keep_step", type=int, default=400000) # Changed from 100000
    # decay_step: how many steps to decay the learning rate for
    parser.add_argument("--decay_step", type=int, default=400000) # Changed from 100000
    # shuffle: shuffle the input data
    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')

    opt = parser.parse_args()
    return opt



if __name__ == "__main__":
    main()

   

