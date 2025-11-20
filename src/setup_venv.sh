#!/bin/bash
# Commands to setup a new conda environment and install all the necessary packages
# See the environment.yaml file for "conda env export > environment.yaml" after running this.

set -e

# conda create -n mindeye python=3.10.8 -y
# conda activate mindeye
# conda install numpy matplotlib tqdm scikit-image jupyterlab -y
# conda install -c conda-forge accelerate -y
module load python/3.12.1
python3 -m venv mindeye
source mindeye/bin/activate
pip install --upgrade pip setuptools wheel

#pip install opencv-python-headless==4.6.0.66 --only-binary :all:
pip install -c constraints.txt numpy matplotlib tqdm scikit-image jupyterlab

pip install -c constraints.txt accelerate

pip install clip-retrieval webdataset clip pandas matplotlib ftfy regex kornia umap-learn
pip install dalle2-pytorch

pip install -c constraints.txt torchvision==0.15.2 torch==2.0.1
pip install -c constraints.txt diffusers==0.13.0

pip install -c constraints.txt info-nce-pytorch==0.1.0
pip install -c constraints.txt pytorch-msssim
