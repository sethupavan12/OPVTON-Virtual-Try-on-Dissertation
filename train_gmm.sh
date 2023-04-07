#!/bin/bash --login
### 

# Request 4 V100 GPUs
#SBATCH -p gpu,gpu_v100 --gres=gpu:1

#job name
#SBATCH --job-name=trainGMM_WITH_PERP_LOSS_0.5

#job stdout file
#SBATCH --output=bench.out.%J

#SBATCH -n 30

#job stderr file
#SBATCH --error=bench.err.%J

module load python/3.9.2
module load CUDA
module load compiler/gnu/9/2.0

cd "/scratch/c.c1984628/my_diss"

# activate the environment
source myenv/bin/activate

pip3 install piqa
pip3 install optuna
pip3 install pytorch-msssim


python3 /scratch/c.c1984628/my_diss/train_bpgm_hyper.py --checkpoint_dir "/scratch/c.c1984628/my_diss/checkpoints/new_ssim_loss_new_channels_higher_ssim_weight" --name "GMM_with_SSIM_loss_new_channels_higher_weight"