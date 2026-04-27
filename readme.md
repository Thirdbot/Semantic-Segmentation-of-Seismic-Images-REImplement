# Implementation of Semantic Segmentation Of Seismic Image

## from: https://arxiv.org/pdf/1905.04307
## using: rockML
## env: Conda

## support only: tensorflow >= 2.10 <= 2.16

# Installations
```bash
conda env create -f environment.yml

conda activate rockml

git clone git@github.com:IBM/rockml.git

pip install ./rockml 
```

The environment file intentionally lists only direct dependencies and version
ranges. Avoid exporting a full lock file with transitive packages when sharing
across Linux, macOS, and Windows because platform-specific wheels can make the
install fail on a different OS.

# test installations
```bash
cd rockml

pytest tests
```

