import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys

sys.path.append('../')
import multiprocessing as mp
import os
import argparse
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from bpgm.model.models import BPGM
from bpgm.model.utils import load_checkpoint, save_checkpoint
from bpgm.dataset import DataLoader, VitonDataset
from torchvision import transforms
from torchvision.utils import save_image


def get_opt():
    parser = argparse.ArgumentParser()
    # Name of the GMM or TOM model
    parser.add_argument("--name", default="testu") # NEEDED
    parser.add_argument("--workers", type=int, default=mp.cpu_count() // 2)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument("--dataroot", default="/scratch/c.c1984628/my_diss/bpgm/data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="/scratch/c.c1984628/my_diss/bpgm/data/train_pairs.txt")
    parser.add_argument("--dataset", default="viton")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str,
                        default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='model checkpoint for initialization') # NEEDED
    parser.add_argument("--display_count", type=int, default=20)

    parser.add_argument("--save_count", type=int, default=5000)

    parser.add_argument("--keep_step", type=int, default=100000)

    parser.add_argument("--decay_step", type=int, default=100000)

    parser.add_argument("--shuffle", action='store_true',
                        help='shuffle input data')
    parser.add_argument("--source_image", type=str, default="000095_0.jpg") # NEEDED
    parser.add_argument("--target_image", type=str, default="000096_0.jpg") # NEEDED
    parser.add_argument("--tom_checkpoint", type=str, default="/scratch/c.c1984628/my_diss/bpgm/checkpoints/TOM/TOM.pth")

    opt = parser.parse_args()
    return opt


# a function that will find the index of the image in the dataset.filepath_df given the name of the image
def find_index(dataset, name):
    for i in range(len(dataset.filepath_df)):
        if dataset.filepath_df.iloc[i, 0] == name:
            return i
    return -1

def generate_of_gmm_output(model, dataset, opt):
    

    for i in range(len(dataset.filepath_df)):
        print("Processing image: ", i)

        given_image = opt.source_image
        target_image = opt.target_image

        given_index = find_index(dataset, given_image)
        print("Given index: ", given_index)
        if given_index == -1:
            print("Given Image not found")
            return
        
        target_index = find_index(dataset, target_image)
        print("Target index: ", target_index)
        if target_index == -1:
            print("Target Image not found")
            return
        
        images = dataset[target_index]
        images_swap = dataset[given_index]
        

        # images_swap = dataset[0]
        print("Name of the source image: ", images_swap['im_name'])
        print("Name of the target image: ", images['im_name'])
        

        tc = images['target_cloth'].unsqueeze(0).cuda()
        tc_for_save = images['target_cloth']
        tc_for_save = tc_for_save / 2 + 0.5
        tc_for_save = tc_for_save.permute(1, 2, 0).numpy()
        tc_for_save = (tc_for_save * 255).astype(np.uint8)


        # Load the original image
        im = Image.open(os.path.join("/scratch/c.c1984628/my_diss/bpgm/data/image", images_swap['im_name']))
        # save the original image to the sample folder
        im.save(os.path.join("/scratch/c.c1984628/my_diss/testing_bpgm", "given.png"))

        # save target cloth to the sample folder
        im = Image.fromarray(tc_for_save).save(f"/scratch/c.c1984628/my_diss/testing_bpgm/{opt.name}/target_cloth_original.png")

        


        # im = Image.fromarray(warped_cloth).save("/scratch/c.c1984628/my_diss/testing_bpgm/original/warped_cloth_original.png")
        #im = Image.fromarray(warped_cloth_masked).save("/scratch/c.c1984628/my_diss/bpgm/results/sample/warped_cloth_masked.png")
        # im = Image.fromarray(warped_mask).save("/scratch/c.c1984628/my_diss/testing_bpgm/original/warped_mask_original.png")
        
        # DEAL WITH SWAP
        tc = images_swap['target_cloth'].unsqueeze(0).cuda()
        tcm = images_swap['target_cloth_mask'].unsqueeze(0).cuda()
        im_bm = images['body_mask'].unsqueeze(0).cuda()
        im_label = images['body_label'].unsqueeze(0).cuda()
        # agnostic = images['agnostic'].unsqueeze(0).cuda()
        
        grid = model(im_label, tc)
        # grid = model(agnostic, tc)
        
        warped_cloth_swap = F.grid_sample(tc, grid, padding_mode='border', align_corners=True)
        
        warped_cloth_masked_swap = warped_cloth_swap * im_bm
        warped_mask_swap = F.grid_sample(tcm, grid, padding_mode='border', align_corners=True)
        
        warped_cloth_swap = warped_cloth_swap.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth_swap = (warped_cloth_swap * 255).astype(np.uint8)
        
        warped_cloth_masked_swap = warped_cloth_masked_swap.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth_masked_swap = (warped_cloth_masked_swap * 255).astype(np.uint8)
        
        warped_mask_swap = warped_mask_swap.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_mask_swap = np.repeat(warped_mask_swap, 3, axis=-1)
        warped_mask_swap = (warped_mask_swap * 255).astype(np.uint8)

        # load the target image
        im = Image.open(os.path.join("/scratch/c.c1984628/my_diss/bpgm/data/image", images['im_name']))
        # save the source image
        
        # save the target image to the sample folder
        #  make a dir for the target image
        # os.mkdir(os.path.join("/scratch/c.c1984628/my_diss/virtual_try_on_results/", opt.name))
        im.save(os.path.join(f"/scratch/c.c1984628/my_diss/virtual_try_on_results/{opt.name}", "target.png"))

        im = Image.open(os.path.join("/scratch/c.c1984628/my_diss/bpgm/data/image", images_swap['im_name']))
        im.save(os.path.join(f"/scratch/c.c1984628/my_diss/virtual_try_on_results/{opt.name}", "source.png"))
        
        im = Image.fromarray(warped_cloth_swap).save(f"/scratch/c.c1984628/my_diss/virtual_try_on_results/{opt.name}/viton_bpgm_warp_original.png")
        return (f"/scratch/c.c1984628/my_diss/virtual_try_on_results/{opt.name}/viton_bpgm_warp_original.png",target_index)
        



def main():

    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)
   
    if opt.dataset == "viton":
        dataset = VitonDataset(opt)
    else:
        raise NotImplementedError
    
    bpgm_model = BPGM(opt)
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(bpgm_model, opt.checkpoint)
    else:
        raise NotImplementedError
    
    bpgm_model.cuda()
    bpgm_model.eval()

    output_dir = f'/scratch/c.c1984628/my_diss/virtual_try_on_results/{opt.name}'
    os.makedirs(output_dir, exist_ok=True)



    (warped_cloth_location,target_index) = generate_of_gmm_output(bpgm_model, dataset, opt)

    # Replace with the path to your saved checkpoint
    # checkpoint_path = '/scratch/c.c1984628/my_diss/checkpoints/TOM_with_CPVTON_LIKE_BUT_GAN/tom_final.pth'
    model = VirtualTryOnUNet().cuda()
    if opt.checkpoint == '':
        print("No checkpoint given!")
        raise NotImplementedError
    model.load_state_dict(torch.load(opt.tom_checkpoint))
    model.eval()
    train_loader = DataLoader(opt, dataset)
    inputs = train_loader.next_batch()

    target = dataset[target_index]

    agnostic = target['body_image'].cuda().unsqueeze(0)  # Person wearing cloth B

    #c = target['parse_body'].cuda().unsqueeze(0)  # Cloth A

    # take a picture from a path and convert it to a tensor
    image= Image.open(warped_cloth_location)
    c = transforms.ToTensor()(image)
    c = c.cuda().unsqueeze(0)

    cm = target['cloth_mask'].cuda().unsqueeze(0)
    seg = target['body_label'].cuda().unsqueeze(0)


    # # Run the model on the input
    # with torch.no_grad():
    #     p_rendered, m_composite = model(torch.cat([agnostic, c, cm], 1))

    # # Compute p_tryon
    # p_tryon = c * m_composite + p_rendered * (1 - m_composite)

    p_rendered, m_composite = model(torch.cat([agnostic, c, cm,seg], 1))

    # Compute p_tryon
    p_tryon = c * m_composite + p_rendered * (1 - m_composite)




    # Save the output image

    # decrease brightness and contrast of the tryon result
    # brightness and contrast are multiplied by 0.5
    # p_tryon = p_tryon * 0.5

    save_image(p_tryon, os.path.join(output_dir, 'tryon_result.png'))

    # also save cloth and im_0
    save_image(c, os.path.join(output_dir, 'cloth.png'))
    save_image(agnostic, os.path.join(output_dir, 'agnostic.png'))

################################################ PASTE GENERATOR ARCHITECTURE HERE ################################################

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CPVTON LIKE BUT GAN ARCHITECTURE START %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CPVTON LIKE BUT GAN ARCHITECTURE END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if __name__ == '__main__':
    main()