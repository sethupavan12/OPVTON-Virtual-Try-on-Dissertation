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
from torchvision import transforms
# import save_image
from torchvision.utils import save_image

# Model used for tom_with_adv_loss
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.conv = nn.Conv2d(7, 3, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# model used for tom_with_adv_loss_deeper_network
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x


# Model used for TOM_with_adversarial_loss_complex_with_seg_higher_batch_size_16 it takes seg as well
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


def main():
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)

    # create dataset
    if opt.dataset == "viton":
        dataset = VitonDataset(opt)
    else:
        raise NotImplementedError

    # create dataloader
    train_loader = DataLoader(opt, dataset)

        
    # Replace with the path to your saved checkpoint
    checkpoint_path = '/scratch/c.c1984628/my_diss/checkpoints/TOM_with_adversarial_loss_complex_with_seg_higher_batch_size_16/tom_adv_loss_final.pth'

    # Load the model from the checkpoint
    model = Generator().cuda()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    inputs = train_loader.next_batch()

    # Image you want to put new clothes on

    # given_image = "000003_0.jpg"

    # # Cloth you want to put on the image
    # given_cloth = "000004_1.jpg"
    # Get the index of the image and cloth in the dataset

    #source = dataset[33]
    target = dataset[48]

    # Use BPGM to generate the warped cloth and mask


    agnostic = target['body_image'].cuda().unsqueeze(0)  # Person wearing cloth B

    #c = target['parse_body'].cuda().unsqueeze(0)  # Cloth A

    # take a picture from a path and convert it to a tensor
    image= Image.open('/scratch/c.c1984628/my_diss/testing_bpgm/original/viton_bpgm_warp_original.png')
    c = transforms.ToTensor()(image)
    c = c.cuda().unsqueeze(0)

    cm = target['cloth_mask'].cuda().unsqueeze(0)
    seg = target['body_label'].cuda().unsqueeze(0)


    # # Run the model on the input
    # with torch.no_grad():
    #     p_rendered, m_composite = model(torch.cat([agnostic, c, cm], 1))

    # # Compute p_tryon
    # p_tryon = c * m_composite + p_rendered * (1 - m_composite)

    p_tryon = model(torch.cat([agnostic, c, cm,seg], 1))




    # Save the output image
    output_dir = '/scratch/c.c1984628/my_diss/tom/testing_tom/'
    os.makedirs(output_dir, exist_ok=True)
    # decrease brightness and contrast of the tryon result
    # brightness and contrast are multiplied by 0.5
    p_tryon = p_tryon * 0.5

    save_image(p_tryon, os.path.join(output_dir, 'tryon_result_with_old_GMM.png'))

    # also save cloth and im_0
    save_image(c, os.path.join(output_dir, 'cloth.png'))
    save_image(agnostic, os.path.join(output_dir, 'agnostic.png'))

def find_index(dataset, name):
    print("Len of dataset: ", len(dataset.filepath_df))
    print(" Looking for ", name, " in dataset at path: ", dataset.filepath_df.iloc[0, 0])
    for i in range(len(dataset.filepath_df)):
        if dataset.filepath_df.iloc[i, 0] == name:
            return i
    return -1   

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
    parser.add_argument('-b', '--batch-size', type=int, default=2)

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


if __name__ == "__main__":
    main()
