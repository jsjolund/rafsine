#!/bin/bash

sudo apt-get update

sudo apt-get upgrade

sudo apt-get -y install ccache clang-format libluajit-5.1-common libluajit-5.1-dev liblua5.1-0 liblua5.1-0-dev luarocks libboost-all-dev nvidia-driver-390 nvidia-utils-390 nvidia-cuda-dev nvidia-cuda-gdb nvidia-cuda-toolkit mesa-utils qt5-default qtbase5-dev xubuntu-desktop geany geany-plugins lightdm lightdm-gtk-greeter xorg cmake clang libgif-dev librsvg2-dev libxine2-dev libpth-dev gdb zsh zsh-syntax-highlighting doxygen graphviz dia mscgen libglm-dev python-pip freeglut3-dev libjpeg9-dev libsdl-dev libsdl2-dev libgstreamer1.0-dev libxml2-dev libcurl4-gnutls-dev libpoppler-cpp-dev libpoppler-glib-dev

# VirtualGL
wget -O virtualgl.deb https://downloads.sourceforge.net/project/virtualgl/2.6.3/virtualgl_2.6.3_amd64.deb
sudo dpkg -i virtualgl.deb
echo "export PATH=\$PATH:/opt/TurboVNC/bin:/opt/VirtualGL/bin:~/.local/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib64" >> ~/.bashdc
source ~/.bashrc

# TurboVNC
wget -O turbovnc.deb https://sourceforge.net/projects/turbovnc/files/2.2.3/turbovnc_2.2.3_amd64.deb/download
sudo dpkg -i turbovnc.deb

# VScode
wget -O vscode.deb https://packages.microsoft.com/repos/vscode/pool/main/c/code/code_1.41.1-1576681836_amd64.deb
sudo dpkg -i vscode.deb
pip install cpplint cmake_format numpy sympy pandas matplotlib scipy
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

# Disable system sleep
sudo apt-get -y remove light-locker xscreensaver
sudo apt autoremove
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target

sudo luarocks install multikey
sudo luarocks install penlight

# Configure VirtualGL
wget https://gist.githubusercontent.com/jsjolund/c783b011e2ea2abee6a8c91de056f3c5/raw/9ce1b031a2b5d5ec7db05da48182535a0f46b144/xorg.conf
sudo mv -f xorg.conf /etc/X11/xorg.conf
sudo systemctl stop lightdm
sudo modprobe -rf nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo vglserver_config -config -s -f -t
sudo usermod -aG vglusers ubuntu
sudo modprobe nvidia_drm nvidia_modeset nvidia_uvm nvidia

sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo update-initramfs -u

sudo systemctl set-default graphical.target

mkdir ~/code

# Install OpenSceneGraph
cd ~/code
git clone https://github.com/openscenegraph/OpenSceneGraph.git
cd OpenSceneGraph
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install

# Rafsine
cd ~/code
git clone https://github.com/jsjolund/rafsine.git
cd rafsine
git config credential.helper store

#Zsh
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

# Set time zone to Europe/Stockholm
sudo ln -fs /usr/share/zoneinfo/Europe/Stockholm /etc/localtime && sudo dpkg-reconfigure --frontend noninteractive tzdata

sudo reboot

# Start the vnc
# ssh -x -e none -L 5903:127.0.0.1:5901 -p 31759 ubuntu@109.225.89.161 -i ~/.ssh/gpusrv_rsa
# vncserver -geometry 1920x1200 -localhost -3dwm -nohttpd -securitytypes tlsnone,x509none,none
# vglrun xfce4-session --display=:1 --screen=0
