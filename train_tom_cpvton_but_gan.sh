#!/bin/bash --login
### 

# Request 4 V100 GPUs
#SBATCH -p gpu,gpu_v100 --gres=gpu:1

#job name
#SBATCH --job-name=train_tom_CPVTON_LIKE_SSIM_LOSS_BUT_GAN

#job stdout file
#SBATCH --output=bench.out.cpvtonLikeSSIMGAN.%J

#SBATCH -n 30

#job stderr file
#SBATCH --error=bench.err.cpvtonLikeSSIMGAN.%J

module load python/3.9.2
module load CUDA
module load compiler/gnu/9/2.0

cd "/scratch/c.c1984628/my_diss"

# activate the environment
source myenv/bin/activate

pip3 install piqa
pip3 install optuna
pip3 install pytorch-msssim

# Run the python script
python3 /scratch/c.c1984628/my_diss/train_tom_cpvton_but_gan.py