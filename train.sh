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

# Run the python script
python3 /scratch/c.c1984628/my_diss/train_bpgm_hyper.py --checkpoint_dir "/scratch/c.c1984628/my_diss/checkpoints/vgg_net_hypertune_bigger_val_data" --name "GMM_with_hyper_tune_bigger_val" --save_count 25000