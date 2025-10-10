#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000M
#SBATCH --time=0-00:20:00
#SBATCH --output=logs_%J.log   

# Load the required modules
module load python/3.11
module load gcc opencv
source /home/psaha03/scratch/env/bin/activate

python /home/psaha03/scratch/cxr-generation/classifier/oversample.py