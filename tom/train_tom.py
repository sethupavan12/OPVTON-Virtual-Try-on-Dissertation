import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time
from unet import CAGUnetGenerator
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


def get_warped_image():
    opt = get_opt()
    opt.train_size = 0.9
    opt.val_size = 0.1
    opt.img_size = 256

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
        
        # we need to get this from ['warped_cloth]
        images = dataset[8855]
        

        images_swap = dataset[12546]
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
        im.save(os.path.join("/scratch/c.c1984628/my_diss/bpgm/results/sample", "original.png"))



        im = Image.fromarray(warped_cloth).save("/scratch/c.c1984628/my_diss/bpgm/results/sample/warped_cloth.png")
        #im = Image.fromarray(warped_cloth_masked).save("/scratch/c.c1984628/my_diss/bpgm/results/sample/warped_cloth_masked.png")
        im = Image.fromarray(warped_mask).save("/scratch/c.c1984628/my_diss/bpgm/results/sample/warped_mask.png")
        
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
        im.save(os.path.join("/scratch/c.c1984628/my_diss/bpgm/results/sample", "target.png"))
        
        im = Image.fromarray(warped_cloth).save("/scratch/c.c1984628/my_diss/warped_cloth.png")
        im = Image.fromarray(warped_cloth_swap).save("/scratch/c.c1984628/my_diss/bpgm/results/sample/viton_bpgm_warp.png")

def train_cag(opt, train_loader, model, board):

    # Make the model use the GPU
    model.cuda()
    # Set the model in training mode
    model.train()

    # Define the loss functions
    criterionL1 = nn.L1Loss()
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for step in range(opt.keep_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        # Get the necessary inputs for CAG
        im_label = inputs['body_label'].cuda()
        
        im_c = inputs['cloth'].cuda()
        im_bm = inputs['body_mask'].cuda()
        im_cm = inputs['cloth_mask'].cuda()

        # IMPLEMENT
        warped_cloth = inputs['warped_cloth'].cuda()


        # Create the input for the CAG
        ic = torch.cat([im_label, warped_cloth, im_c, im_bm, im_cm], dim=1)

        # Generate the output using CAG
        output = model(ic)

        # Calculate the loss
        vgg_p_loss = loss_fn_vgg.forward(output, im_c)
        vgg_p_loss = torch.mean(vgg_p_loss)
        loss_cloth = criterionL1(output, im_c) + 0.5 * vgg_p_loss
        loss_mask = criterionL1(output * im_cm, im_c * im_cm) * 0.1
        loss = loss_cloth + loss_mask

        # Zero the gradients, perform backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % opt.display_count == 0:
            visuals = [[output, im_c, warped_cloth],
                       [output * im_cm, im_c * im_cm, im_cm]]

            board_add_images(board, 'combine', visuals, step+1)
            board.add_scalar('metric', loss.item(), step+1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()), flush=True)

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(logdir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'gmm_final.pth'))
    elif opt.stage == 'TOM':
        # model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)  # CP-VTON
        cag_model = CAGUnetGenerator(input_nc=22, output_nc=3, num_downs=7, ngf=64) # CP-VTON+
        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, train_loader, model, board)
        save_checkpoint(model, os.path.join(
            opt.checkpoint_dir, opt.name, 'tom_final.pth'))
    else:
        raise NotImplementedError('Model [%s] is not implemented' % opt.stage)

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))