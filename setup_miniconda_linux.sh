#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
chmod +x Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
./Miniconda3-py310_23.5.0-3-Linux-x86_64.sh

#source ~/.bashrc
#rm Miniconda3-py310_23.5.0-3-Linux-x86_64.sh
#conda update conda
#conda update -n base conda
#conda install -n base conda-libmamba-solver
#conda config --set solver libmamba

#git config --global user.name "github-username"
#git config --global user.email "email@email.com"
#git clone https://github.com/parlance-zz/dualdiffusion

# *env setup*
#cd dualdiffusion
#conda env create -f environment.yml
#conda activate dualdiffusion
#cp ./.env.default ./.env
#nano ./.env
#accelerate config
# - torch dynamo no
# - deepspeed no
# - mixed precision fp16
# *install python vscode extension and set interpreter to dualdiffusion env*

# *create windows share*
#sudo apt install samba
#sudo nano /etc/samba/smb.conf
# [ShareName]
# path = /home/username/sharepath
# valid users = username
# read only = no
# browsable = yes
#sudo smbpasswd -a username
#sudo service smbd restart
#sudo ufw allow Samba

# *disable x on startup*
#sudo systemctl isolate multi-user.target
#sudo systemctl set-default multi-user.target

# *setup ssh*
#sudo apt install openssh-server
#sudo systemctl enable ssh
#sudo ufw allow ssh
#sudo systemctl start ssh
#ssh-keygen -t rsa -b 8192
# *copy /home/username/.ssh/id_rsa to windows vscode and add new host*
# *copy /home/username/.ssh/id_rsa.pub to github ssh keys*

# *ubuntu install options*
#-disable secure boot in bios
#-enable install 3rd party drivers
#-if sharing hdd with windows:
#  -create 2 unformatted 'raw' partitions in windows
#  -use existing efi partition for boot
#  -use 1st raw partition as ext4 mount /
#  -use 2nd raw partition as swap
#-set default boot OS in bios

# *enable discard/trim for ssd*
#sudo nano /etc/fstab
# *enable discard and noatime: UUID=df39839a-05b1-4bc9-9c5c-3ee6b30981a8 /               ext4    discard,noatime,errors=remount-ro 0*
#reboot