# Install on Ubuntu 20.04

#!/bin/bash

sudo apt-get update

sudo apt-get upgrade

# Install latest CUDA and Nvidia drivers
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
sudo apt update
sudo bash -c 'echo "precedence ::ffff:0:0/96  100" >> /etc/gai.conf'
sudo add-apt-repository ppa:graphics-drivers
sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'

sudo apt update
sudo apt install cuda-11-1

vim ~/.profile

bash -c 'echo "PATH=/opt/TurboVNC/bin:/opt/VirtualGL/bin:~/.local/bin:/usr/local/cuda-11.1/bin${PATH:+:${PATH}}" >> ~/.profile'
bash -c 'echo "LD_LIBRARY_PATH=/usr/lib64:/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.profile'

# VirtualGL
wget -O virtualgl.deb https://downloads.sourceforge.net/project/virtualgl/2.6.5/virtualgl_2.6.5_amd64.deb
sudo dpkg -i virtualgl.deb

# TurboVNC
wget -O turbovnc.deb https://sourceforge.net/projects/turbovnc/files/2.2.5/turbovnc_2.2.5_amd64.deb/download
sudo dpkg -i turbovnc.deb

# Disable system sleep
sudo apt-get -y remove light-locker xscreensaver xfce4-screensaver
sudo apt autoremove
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo apt-get remove update-manager

# Set time zone to Europe/Stockholm
sudo ln -fs /usr/share/zoneinfo/Europe/Stockholm /etc/localtime && sudo dpkg-reconfigure --frontend noninteractive tzdata

# Configure VirtualGL
sudo nvidia-xconfig
# Set xorg.conf BusID "PCI:130:0:0" from nvidia-xconfig --query-gpu-info
sudo modprobe -rf nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo vglserver_config -config -s -f -t
sudo usermod -aG vglusers ubuntu
sudo modprobe nvidia_drm nvidia_modeset nvidia_uvm nvidia

# Disable nouveau drivers
sudo bash -c 'echo -e "blacklist nouveau\noptions nouveau modeset=0" > /etc/modprobe.d/blacklist-nvidia-nouveau.conf'
sudo update-initramfs -u

sudo systemctl set-default graphical.target

# Start the vnc
mkdir ~/.vnc
bash -c 'echo -e "#!/bin/sh\nvglrun xfce4-session --display=:1" > ~/.vnc/xstartup.turbovnc'
chmod +x ~/.vnc/xstartup.turbovnc

sudo bash -c 'echo -e "VNCSERVERS=\"1:ubuntu\"\nVNCSERVERARGS[1]=\"-geometry 1920x1080 -localhost -3dwm -nohttpd -securitytypes tlsnone,x509none,none\"" >> /etc/sysconfig/tvncservers'
sudo update-rc.d tvncserver defaults

# cat "/home/ubuntu/.vnc/$(hostname):1.log"

# ssh -x -e none -L 5902:127.0.0.1:5901 -p 31759 ubuntu@109.225.89.161 -i ~/.ssh/gpusrv_rsa

# Build dependencies
sudo apt-get -y install libluajit-5.1-common libluajit-5.1-dev liblua5.1-0 liblua5.1-0-dev luarocks libboost-all-dev qt5-default qtbase5-dev cmake
sudo luarocks install penlight

# Dev dependencies
sudo apt-get -y install ccache clang clang-format gdb doxygen graphviz dia mscgen python3-pip
pip3 install --user pygments --upgrade
pip3 install --user cpplint cmake_format numpy sympy pandas matplotlib scipy jupyter notebook

# Runtime dependencies and other tools
sudo apt-get -y install mesa-utils xubuntu-desktop geany geany-plugins lightdm lightdm-gtk-greeter xorg

# VScode
wget -O vscode.deb https://packages.microsoft.com/repos/vscode/pool/main/c/code/code_1.51.1-1605051630_amd64.deb
sudo dpkg -i vscode.deb
code --install-extension njpwerner.autodocstring
code --install-extension davidanson.vscode-markdownlint
code --install-extension austin.code-gnu-global
code --install-extension cheshirekow.cmake-format
code --install-extension cschlosser.doxdocgen
code --install-extension eamodio.gitlens
code --install-extension kriegalex.vscode-cudacpp
code --install-extension mechatroner.rainbow-csv
code --install-extension mine.cpplint
code --install-extension mitaki28.vscode-clang
code --install-extension ms-python.python
code --install-extension ms-vscode.cmake-tools
code --install-extension ms-vscode.cpptools
code --install-extension slevesque.vscode-hexdump
code --install-extension twxs.cmake
code --install-extension webfreak.debug
code --install-extension xaver.clang-format
wget -O ~/.config/Code/User/keybindings.json https://gist.githubusercontent.com/jsjolund/fd45ea95b31b35dc9f4d6857b4f97cd4/raw/a0ad291188225d8e6f27e4cbd49456da9b2f679f/keybindings.json
wget -O ~/.config/Code/User/settings.json https://gist.githubusercontent.com/jsjolund/f2bb95d720b1ad2806e259f53caf41f8/raw/4657e45ac2633a1fc215a32b947251b7a71b0bad/settings.json

# Install OpenSceneGraph
sudo apt install libopenscenegraph-dev openscenegraph libopenthreads-dev libopenthreads21

#Zsh
cd ~
zsh
git clone --recursive https://github.com/sorin-ionescu/prezto.git "${ZDOTDIR:-$HOME}/.zprezto"
setopt EXTENDED_GLOB
for rcfile in "${ZDOTDIR:-$HOME}"/.zprezto/runcoms/^README.md(.N); do
ln -s "$rcfile" "${ZDOTDIR:-$HOME}/.${rcfile:t}"
done
sudo chsh -s /bin/zsh ubuntu
echo "export PATH=\$PATH:/opt/TurboVNC/bin:/opt/VirtualGL/bin:~/.local/bin" >> ~/.zpreztorc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib64" >> ~/.zpreztorc
source ~/.zpreztorc


# Rafsine
# cd ~
# git clone https://github.com/jsjolund/rafsine.git
# cd rafsine
# git config credential.helper store

# nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java

# clang-format -i --verbose include/**/*.hpp; clang-format -i --verbose src/**/*.cpp
