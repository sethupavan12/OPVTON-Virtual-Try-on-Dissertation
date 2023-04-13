#!/bin/bash --login
### 

# Request 4 V100 GPUs
#SBATCH -p gpu,gpu_v100,c_gpu_comsc1 --gres=gpu:1

#job name
#SBATCH --job-name=trainTOMWITH_PERPLOSS

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
pip3 install lpips

# Run the python script
python3 /scratch/c.c1984628/my_diss/train_tom_with_perp.py --checkpoint_dir "/scratch/c.c1984628/my_diss/tom/checkpoints/TOM_perp_loss" --name "TOM_perp_loss" --save_count 10000