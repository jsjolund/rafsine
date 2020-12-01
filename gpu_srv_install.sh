#!/bin/bash

sudo apt-get update

sudo apt-get upgrade

# Build dependencies
sudo apt-get -y install libluajit-5.1-common libluajit-5.1-dev liblua5.1-0 liblua5.1-0-dev luarocks libboost-all-dev nvidia-cuda-dev nvidia-cuda-toolkit qt5-default qtbase5-dev cmake libglm-dev
sudo luarocks install multikey
sudo luarocks install penlight

# Dev dependencies
sudo apt-get -y install ccache clang clang-format gdb doxygen graphviz dia mscgen
pip3 install cpplint cmake_format numpy sympy pandas matplotlib scipy jupyter notebook

# Runtime dependencies and other tools
sudo apt-get -y install nvidia-driver-390 nvidia-utils-390 nvidia-cuda-gdb mesa-utils xubuntu-desktop geany geany-plugins lightdm lightdm-gtk-greeter xorg libgif-dev librsvg2-dev libxine2-dev libpth-dev zsh zsh-syntax-highlighting python3-pip freeglut3-dev libjpeg9-dev libsdl-dev libsdl2-dev libgstreamer1.0-dev libxml2-dev libcurl4-gnutls-dev libpoppler-cpp-dev libpoppler-glib-dev

# VirtualGL
wget -O virtualgl.deb https://downloads.sourceforge.net/project/virtualgl/2.6.5/virtualgl_2.6.5_amd64.deb
sudo dpkg -i virtualgl.deb

# TurboVNC
wget -O turbovnc.deb https://sourceforge.net/projects/turbovnc/files/2.2.5/turbovnc_2.2.5_amd64.deb/download
sudo dpkg -i turbovnc.deb

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

# Disable system sleep
sudo apt-get -y remove light-locker xscreensaver
sudo apt autoremove
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo apt-get remove update-manager

# Set time zone to Europe/Stockholm
sudo ln -fs /usr/share/zoneinfo/Europe/Stockholm /etc/localtime && sudo dpkg-reconfigure --frontend noninteractive tzdata

# Configure VirtualGL
wget https://gist.githubusercontent.com/jsjolund/c783b011e2ea2abee6a8c91de056f3c5/raw/9ce1b031a2b5d5ec7db05da48182535a0f46b144/xorg.conf
sudo mv -f xorg.conf /etc/X11/xorg.conf
sudo systemctl stop lightdm
sudo modprobe -rf nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo vglserver_config -config -s -f -t
sudo usermod -aG vglusers ubuntu
sudo modprobe nvidia_drm nvidia_modeset nvidia_uvm nvidia

# Disable nouveau drivers
sudo bash -c "echo blacklist nouveau\necho options nouveau modeset=0 > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u

sudo systemctl set-default graphical.target

# Start the vnc
echo "#\!/bin/sh\nvglrun xfce4-session --display=:1 --screen=0" > ~/.vnc/xstartup.turbovnc
chmod +x ~/.vnc/xstartup.turbovnc

sudo echo "VNCSERVERS=\"1:ubuntu\"\nVNCSERVERARGS[1]=\"-geometry 1920x1080 -localhost -3dwm -nohttpd -securitytypes tlsnone,x509none,none" > /etc/sysconfig/tvncservers
sudo update-rc.d tvncserver defaults

# cat "/home/ubuntu/.vnc/$(hostname):1.log"

# ssh -x -e none -L 5902:127.0.0.1:5901 -p 31759 ubuntu@109.225.89.161 -i ~/.ssh/gpusrv_rsa

# Install OpenSceneGraph
cd ~
git clone https://github.com/openscenegraph/OpenSceneGraph.git
cd OpenSceneGraph
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

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
echo "export PATH=\$PATH:/opt/TurboVNC/bin:/opt/VirtualGL/bin:~/.local/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib64" >> ~/.bashrc
source ~/.bashrc

# Rafsine
# cd ~
# git clone https://github.com/jsjolund/rafsine.git
# cd rafsine
# git config credential.helper store
