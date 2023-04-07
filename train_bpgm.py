#coding=utf-8
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys

sys.path.append('../')

import time
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


def train_bpgm(opt, train_loader, model, board):

    # Make the model use the GPU
    model.cuda()
    # Set the model in training mode
    model.train()

    # Define the loss functions
    # L1 loss
    # L1 loss is the sum of the absolute differences between the predicted and the target values
    #criterionL1 = nn.L1Loss()
    # VGG loss

    # criterionVGG = VGGLoss() # This is the loss function used in the original paper

    # SSIM loss
    #criterionSSIM = SSIMLoss().cuda()
    # loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    #CLASSIC LOSS
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    for step in range(opt.keep_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # cloth of the target person
        tc = inputs['target_cloth'].cuda()
        # cloth mask of the target person
        tcm = inputs['target_cloth_mask'].cuda()
        
        # cloth you want to put on the target person
        im_c =  inputs['cloth'].cuda()
        im_bm = inputs['body_mask'].cuda()
        im_cm = inputs['cloth_mask'].cuda()
        
        im_label = inputs['body_label'].cuda()
        # Generate a grid for warping the cloth onto the label image
        grid = model(im_label, tc)
        # Warp the target cloth onto the label image and mask it
        warped_cloth = F.grid_sample(tc, grid, padding_mode='border', align_corners=True)
        warped_cloth = warped_cloth * im_bm
        warped_mask = F.grid_sample(tcm, grid, padding_mode='border', align_corners=True)
        
        # Calculate the loss
        # perceptual loss between warped_cloth and cloth
        #vgg_p_loss = loss_fn_vgg.forward(warped_cloth, im_c)
        # convert vgg_p_loss to a scalar
        # vgg_p_loss = torch.mean(vgg_p_loss)
        # loss_cloth = criterionL1(warped_cloth, im_c) + 0.5 * vgg_p_loss 
        # loss_mask = criterionL1(warped_mask, im_cm) * 0.1
        # loss = loss_cloth + loss_mask

        # CLASSIC LOSS
        loss_cloth = criterionL1(warped_cloth, im_c) + 0.1 * criterionVGG(warped_cloth, im_c)
        loss_mask = criterionL1(warped_mask, im_cm) * 0.1
        loss = loss_cloth + loss_mask
        
        # Zero the gradients, perform backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            label = inputs['label'].cuda()
            im_g = inputs['grid_image'].cuda()
            with torch.no_grad():
                warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)
            
            visuals = [[label, warped_grid, -torch.ones_like(label)], 
                    [tc, warped_cloth, im_c],
                    [tcm, warped_mask, im_cm]]
            
            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

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

def main():
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))
   
    # create dataset 
    # if opt.dataset == "mpv":
    #     train_dataset = MPVDataset(opt)
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
   
    # create model & train & save the final checkpoint
    model = BPGM(opt)
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
        load_checkpoint(model, opt.checkpoint)
        
    train_bpgm(opt, train_loader, model, board)
    save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'bpgm_final.pth'))
  
    print('Finished training %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
