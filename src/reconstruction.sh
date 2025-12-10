#!/bin/bash
#
#SBATCH --job-name=reconstruction_mindeye
#SBATCH --output=/oak/stanford/groups/anishm/gtyagi/reconstruction/reconstruction.%j.out
#SBATCH --error=/oak/stanford/groups/anishm/gtyagi/reconstruction/reconstruction.%j.err
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH -p owners 
#SBATCH -G 1
##SBATCH -C GPU_MEM:24GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mem=64G
#SBATCH -n 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gtyagi@stanford.edu

# Environment setup
source ~/.bashrc
#source mindeye/bin/activate
conda activate mindeye_conda  # Update this to your conda environment name
export PYTHONPATH=""
cd /oak/stanford/groups/anishm/gtyagi/mindeye/fMRI-reconstruction-NSD/src  # Update path to your project directory

# Thread settings for CPU operations
N=8
export OMP_NUM_THREADS=${N}
export OPENBLAS_NUM_THREADS=${N}
export MKL_NUM_THREADS=${N}
export VECLIB_MAXIMUM_THREADS=${N}
export NUMEXPR_NUM_THREADS=${N}

# PyTorch CUDA memory settings
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Run training with accelerate launch (single GPU)
accelerate launch --num_processes 1 Reconstructions.py \
    --model_name="mindeye_test1" \
    --data_path="/oak/stanford/groups/anishm/fMRI_datasets/NSD" \
    --subj=1 \
    --img2img_strength=1 \
    --vd_cache_dir="/oak/stanford/groups/anishm/fMRI_datasets/NSD/vd_cache" \
    --recons_per_sample=4