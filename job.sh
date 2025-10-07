#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-12:00:00
#SBATCH --output=logs_%J.log   

# Load the required modules
module load python/3.11
module load gcc opencv
source /home/psaha03/scratch/env/bin/activate
pip install -r /home/psaha03/scratch/chest-xray-classification/requirements.txt    

python /home/psaha03/scratch/chest-xray-classification/train.py