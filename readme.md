# Implementation of Semantic Segmentation Of Seismic Image

## from: https://arxiv.org/pdf/1905.04307
## using: rockML
## env: Conda

## support only: tensorflow >= 2.10 <= 2.16

# Installations
```bash
conda env create -f environment-gpu-linux.yml

conda activate rockml

git clone git@github.com:IBM/rockml.git

pip install ./rockml 
```


The GPU file uses `tensorflow[and-cuda]` so pip installs TensorFlow's matching
CUDA/cuDNN runtime wheels. This avoids relying on a system CUDA toolkit version.

# test installations
```bash
cd rockml

pytest tests
```
