#!/bin/bash
#
#SBATCH --job-name=train_mindeye
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/jobs/train_mindeye.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/jobs/train_mindeye.%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH -p owners 
#SBATCH -G 1
##SBATCH -C GPU_MEM:24GB
#SBATCH -C GPU_SKU:H100_SXM5
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH --mail-type=END, FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# Environment setup
source ~/.bashrc
source mindeye/bin/activate
#conda activate mindeye  # Update this to your conda environment name
cd /oak/stanford/groups/anishm/gtyagi/mindeye/fMRI-reconstruction-NSD/src  # Update path to your project directory

# Thread settings for CPU operations
N=8
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# PyTorch CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Run training with accelerate launch (single GPU)
accelerate launch --num_processes 1 Train_MindEye.py \
    --model_name="mindeye_test1" \
    --subj=1 \
    --hidden \
    --clip_variant=ViT-L/14 \
    --data_path="/oak/stanford/groups/anishm/fMRI_datasets/NSD" \
    --vd_cache_dir="/oak/stanford/groups/anishm/fMRI_datasets/NSD/vd_cache"

