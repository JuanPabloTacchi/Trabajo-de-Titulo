# Install Miniconda in ~/miniconda3
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Activate Miniconda
source ~/miniconda3/bin/activate

# Create environment and activate it
cd main/Uni3D
conda create -y -n uni3d python=3.8
conda activate uni3d

# Install PyTorch and dependencies without confirmation
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install Uni3D requirements
pip install -r requirements.txt

# Install PointNet2
# For some reason it cannot save the wheel file due to permission issues,
# so we compile it on the fly instead
conda install -y nvidia/label/cuda-11.8.0::cuda-toolkit
cd main/Pointnet2_PyTorch/pointnet2_ops_lib
pip install -e .
cd main/Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .

# Return to the Uni3D folder
cd main/Uni3D