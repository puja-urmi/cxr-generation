#!/bin/bash
#SBATCH --job-name=cxc_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00

# Load the required modules
module load python/3.11
module load gcc opencv

# Activate environment
source /home/psaha03/scratch/env/bin/activate

# Install requirements 
pip install -r /home/psaha03/scratch/cxr-generation/classifier/requirements.txt

# Change to working directory
cd /home/psaha03/scratch/cxr-generation/classifier

# Create output directory with timestamp
OUTPUT_DIR="./test_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python evaluate.py \
    --mode test \
    --model /home/psaha03/scratch/classifier_results/models/densenet_classifier_best.pt \
    --data /home/psaha03/scratch/chest_xray_data/test \
    --output "$OUTPUT_DIR"

echo "Evaluation completed. Results saved to: $OUTPUT_DIR"