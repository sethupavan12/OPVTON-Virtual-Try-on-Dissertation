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

from bpgm.model.models import BPGM
from bpgm.model.utils import load_checkpoint, save_checkpoint
from bpgm.dataset import DataLoader, VitonDataset


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
    parser.add_argument("--stage", default="GMM")
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


# a function that will find the index of the image in the dataset.filepath_df given the name of the image
def find_index(dataset, name):
    for i in range(len(dataset.filepath_df)):
        if dataset.filepath_df.iloc[i, 0] == name:
            return i
    return -1

def main():
    
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    # if opt.dataset == "mpv":
    #     dataset = MPVDataset(opt)
    # el
    if opt.dataset == "viton":
        dataset = VitonDataset(opt)
    else:
        raise NotImplementedError
    
    model = BPGM(opt)
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
    else:
        raise NotImplementedError
    
    model.cuda()
    model.eval()

    for i in range(len(dataset.filepath_df)):
        print("Processing image: ", i)

        # images = dataset[i]
        # images_swap = dataset[i]
        
        # if images['im_name'] != "013418_0.jpg":
        #     continue
        
        # images = dataset[1]
        given_image = "000272_0.jpg"
        target_image = "000288_0.jpg"


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
        
        for key, im in images.items():
            if isinstance(im, torch.Tensor) and im.shape[0] in {1, 3}:
                im = im / 2 + 0.5
                im = im.permute(1, 2, 0).numpy()
                im = (im * 255).astype(np.uint8)
                
                if im.shape[-1] == 1:
                    im = np.repeat(im, 3, axis=-1)
                
                im = Image.fromarray(im)
                # im.save(os.path.join("sample", "bpgm_warp", key + ".png"))

        # DEAL WITH ORIGINAL
        tc = images['target_cloth'].unsqueeze(0).cuda()
        tcm = images['target_cloth_mask'].unsqueeze(0).cuda()
        im_bm = images['body_mask'].unsqueeze(0).cuda()
        im_label = images['body_label'].unsqueeze(0).cuda()
        # agnostic = images['agnostic'].unsqueeze(0).cuda()
            
        grid = model(im_label, tc)
        # grid = model(agnostic, tc)
        
        warped_cloth = F.grid_sample(tc, grid, padding_mode='border', align_corners=True)
        
        warped_cloth_masked = warped_cloth * im_bm
        warped_mask = F.grid_sample(tcm, grid, padding_mode='border', align_corners=True)
        
        warped_cloth = warped_cloth.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth = (warped_cloth * 255).astype(np.uint8)
        
        warped_cloth_masked = warped_cloth_masked.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_cloth_masked = (warped_cloth_masked * 255).astype(np.uint8)
        
        warped_mask = warped_mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        warped_mask = np.repeat(warped_mask, 3, axis=-1)
        warped_mask = (warped_mask * 255).astype(np.uint8)
        
        # Load the original image
        im = Image.open(os.path.join("/scratch/c.c1984628/my_diss/bpgm/data/image", images_swap['im_name']))
        # save the original image to the sample folder
        im.save(os.path.join("/scratch/c.c1984628/my_diss/testing_bpgm", "given.png"))



        im = Image.fromarray(warped_cloth).save("/scratch/c.c1984628/my_diss/testing_bpgm/original/warped_cloth_original.png")
        #im = Image.fromarray(warped_cloth_masked).save("/scratch/c.c1984628/my_diss/bpgm/results/sample/warped_cloth_masked.png")
        im = Image.fromarray(warped_mask).save("/scratch/c.c1984628/my_diss/testing_bpgm/original/warped_mask_original.png")
        
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
        # save the target image to the sample folder
        im.save(os.path.join("/scratch/c.c1984628/my_diss/testing_bpgm", "target.png"))
        
        im = Image.fromarray(warped_cloth).save("/scratch/c.c1984628/my_diss/testing_bpgm/original/warped_cloth_original.png")
        im = Image.fromarray(warped_cloth_swap).save("/scratch/c.c1984628/my_diss/testing_bpgm/original/viton_bpgm_warp_original.png")
        #im = Image.fromarray(warped_cloth_swap).save(os.path.join("tmp.jpg"))
        break
        
        # im = Image.fromarray(warped_cloth_masked_swap).save(os.path.join("sample", "bpgm_warp", "warped_cloth_masked_swap.png"))
        # im = Image.fromarray(warped_mask_swap).save(os.path.join("sample", "bpgm_warp", "warped_mask_swap.png"))


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
    parser.add_argument("--stage", default="GMM")
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
