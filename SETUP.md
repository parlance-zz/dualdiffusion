# setup notes - setup script is tbd

# miniconda3 setup
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
chmod +x Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
./Miniconda3-py310_23.5.0-3-Linux-x86_64.sh

source ~/.bashrc
rm Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
conda update conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# git setup
sudo apt install git
git config --global user.name "github-username"
git config --global user.email "email@email.com"
git clone https://github.com/parlance-zz/dualdiffusion

# environment setup
cd dualdiffusion
conda env create -f environment.yml
# alternatively to use a specific location for the conda env
# you could use conda env create -f environment.yml --prefix /path/to/env
conda activate dualdiffusion
cp ./.env.default ./.env
nano ./.env
accelerate config
# - torch dynamo no
# - deepspeed no
# - mixed precision bf16
# *install python vscode extension and set interpreter to dualdiffusion env*

# dataset config
nano ./config/dataset/dataset.json

# dataset upload / download
# huggingface-cli download hf_user/dataset_repo --repo-type dataset --local-dir /path/to/dataset
# huggingface-cli upload hf_user/dataset_repo --repo-type dataset --local-dir /path/to/dataset

# new model setup / training config setup
# copy ./config/default to ./config/model_name
# edit config files as needed
# run create_new_model.py and enter model_name

# ***system setup***

# *ubuntu install / setup*
# disable secure boot in bios
# enable install 3rd party drivers
# if sharing hdd with windows:
#  -create 2 unformatted 'raw' partitions in windows
#  -use existing efi partition for boot
#  -use 1st raw partition as ext4 mount /
#  -use 2nd raw partition as swap
# set default boot OS in bios

# setup ssh
sudo apt install openssh-server
sudo systemctl enable ssh
sudo ufw allow ssh
sudo systemctl start ssh
ssh-keygen -t rsa -b 8192
# copy /home/username/.ssh/id_rsa to windows vscode and add new host
# copy /home/username/.ssh/id_rsa.pub to github ssh keys

# enable discard/trim and noatime for ssd
sudo nano /etc/fstab
# UUID=df39839a-05b1-4bc9-9c5c-3ee6b30981a8 /               ext4    discard,noatime,nodiratime,errors=remount-ro 0
reboot

# disable x on startup
sudo systemctl isolate multi-user.target
sudo systemctl set-default multi-user.target

# create windows share
#sudo apt install samba
#sudo nano /etc/samba/smb.conf
# [ShareName]
# path = /home/username/sharepath
# valid users = username
# read only = no
# browsable = yes
sudo smbpasswd -a username
sudo service smbd restart
sudo ufw allow Samba