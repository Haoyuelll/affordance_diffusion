set -e 
set -x

create -n adiff python=3.9 -y
conda activate adiff

conda install pytorch==2.5 torchvision==0.20 -c pytorch -c conda-forge -y

pip install -r requirements.txt

git clone git@github.com:facebookresearch/pytorch3d.git
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..
