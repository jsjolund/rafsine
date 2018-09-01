#!/bin/bash

PACKAGES=(
luajit
lua5.2-dev
liblua5.1-0-dev
luarocks
libboost-all-dev
nvidia-driver-390
nvidia-utils-390
nvidia-cuda-dev
nvidia-cuda-gdb
nvidia-cuda-toolkit
mesa-utils
qt5-default
qtbase5-dev
xfce4
xfce4-goodies
geany
lightdm
xorg
cmake
clang
libgif-dev
librsvg2-dev
libxine2-dev
libpth-dev
gdb
zsh
)
PKG_STR=$(IFS=$'\n'; echo "${PACKAGES[*]}")

sudo apt-get -y install $(IFS=$'\n'; echo "${PACKAGES[*]}")

# VirtualGL
wget -O virtualgl.deb https://downloads.sourceforge.net/project/virtualgl/2.6/virtualgl_2.6_amd64.deb
sudo dpkg -i virtualgl.deb
echo "export PATH=\$PATH:/opt/TurboVNC/bin:/opt/VirtualGL/bin" >> ~/.bashrc
echo "export TVNC_WM='vglrun xfce4-session --display=:1 --screen=0'" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64" >> ~/.bashrc
source ~/.bashrc

# TurboVNC
wget -O turbovnc.deb https://sourceforge.net/projects/turbovnc/files/2.1.90%20%282.2beta1%29/turbovnc_2.1.90_amd64.deb/download
sudo dpkg -i turbovnc.deb

# VScode
wget -O vscode.deb https://packages.microsoft.com/repos/vscode/pool/main/c/code/code_1.26.1-1534444688_amd64.deb
sudo dpkg -i vscode.deb
code --install-extension ms-vscode.cpptools 
code --install-extension eamodio.gitlens
code --install-extension mitaki28.vscode-clang
code --install-extension austin.code-gnu-global
code --install-extension twxs.cmake
code --install-extension vector-of-bool.cmake-tools
code --install-extension maddouri.cmake-tools-helper
code --install-extension webfreak.debug
code --install-extension kriegalex.vscode-cudacpp
# Disable system sleep
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

sudo luarocks install multikey
sudo luarocks install penlight

#reboot

# Configure VirtualGL
wget https://gist.githubusercontent.com/jsjolund/c783b011e2ea2abee6a8c91de056f3c5/raw/3f14be12fcf6d3a9fa14ea1aa9e054bb988697c7/xorg.conf
sudo mv -f xorg.conf /etc/X11/xorg.conf
sudo systemctl stop lightdm
sudo modprobe -rf nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo vglserver_config -config -s -f -t
sudo usermod -aG vglusers ubuntu
sudo modprobe nvidia_drm nvidia_modeset nvidia_uvm nvidia

#reboot?

# Start the vnc
#ssh -x -e none -L 5903:127.0.0.1:5901 -p 31759 ubuntu@109.225.89.161 -i ~/.ssh/gpusrv_rsa
vncserver -geometry 1920x1200 -localhost -3dwm -nohttpd -securitytypes tlsnone,x509none,none
vglrun xfce4-session --display=:1 --screen=0

# Install OpenSceneGraph
git clone https://github.com/openscenegraph/OpenSceneGraph.git
cd OpenSceneGraph
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release
make -j40

# Rafsine
git clone https://github.com/jsjolund/rafsine-gui.git
cd rafsine-gui
git submodule update --init --recursive
git config credential.helper store

#Zsh
zsh
git clone --recursive https://github.com/sorin-ionescu/prezto.git "${ZDOTDIR:-$HOME}/.zprezto"
setopt EXTENDED_GLOB
for rcfile in "${ZDOTDIR:-$HOME}"/.zprezto/runcoms/^README.md(.N); do
  ln -s "$rcfile" "${ZDOTDIR:-$HOME}/.${rcfile:t}"
done
sudo chsh -s /bin/zsh ubuntu
