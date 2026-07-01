#!/usr/bin/env bash

# Following https://github.com/robocasa/robocasa-gr1-tabletop-tasks

apt install libegl1-mesa-dev libglu1-mesa git-lfs

# 1. Set up conda environment
conda create -c conda-forge -n robocasa python=3.10
conda activate robocasa
pip install websockets

# # 2. Clone and install Isaac-GR00T
# git clone https://github.com/NVIDIA/Isaac-GR00T.git
# cd Isaac-GR00T
# pip install --upgrade setuptools
# pip install -e .[base] # NOTE: remove the flash-attn from `pyproject.toml` and manually download and install flash-attn
# pip install --no-build-isolation flash-attn==2.7.1.post4 
# cd ..


# 3. Clone and install robosuite
# git clone https://github.com/ARISE-Initiative/robosuite.git # NOTE: remove "mujoco>=3.3.0" from `setup.py`
# cd robosuite
# git checkout 1a8701b90c07c6595ace4af9935d7c5ebe1baed3
# pip install -e .
# cd ..

# 3. Clone and install robosuite
wget https://github.com/ARISE-Initiative/robosuite/archive/refs/tags/v1.5.1.zip -O robosuite-1.5.1.zip
unzip robosuite-1.5.1.zip
mv robosuite-1.5.1 robosuite
pip install -e robosuite

# 4. Clone and install robocasa-gr1-tabletop-tasks
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git
pip install -e robocasa-gr1-tabletop-tasks
pip install numpy==1.26.4 imageio[ffmpeg]

# 5. Download assets
cd robocasa-gr1-tabletop-tasks
python robocasa/scripts/download_tabletop_assets.py -y