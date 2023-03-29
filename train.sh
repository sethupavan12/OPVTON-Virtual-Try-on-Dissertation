#!/bin/bash --login
### 

# Request 4 V100 GPUs
#SBATCH -p gpu_v100 --gres=gpu:2

#job name
#SBATCH --job-name=trainGMM

#job stdout file
#SBATCH --output=bench.out.%J

#SBATCH -n 40

#job stderr file
#SBATCH --error=bench.err.%J

module load python/3.9.2
module load compiler/gnu/9/2.0

cd "/scratch/c.c1984628/my_diss"

# activate the environment
source myenv/bin/activate

# Run the python script
python3 /scratch/c.c1984628/my_diss/train_bpgm.py 